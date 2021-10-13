from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import torch.nn as nn
import torch.optim as optim

import numpy as np

import time
import os
from six.moves import cPickle

import opts
import models
from dataloader import *
import eval_utils
import misc.utils as utils
from misc.rewards import init_scorer, get_self_critical_reward

try:
    import tensorboardX as tb
except ImportError:
    print("tensorboardX is not installed")
    tb = None

def add_summary_value(writer, key, value, iteration):
    if writer:
        writer.add_scalar(key, value, iteration)

def train(opt):
    # Deal with feature things before anything
    opt.use_att = utils.if_use_att(opt.caption_model)
    ac = 0

    loader = DataLoader(opt)
    opt.vocab_size = loader.vocab_size
    opt.seq_length = loader.seq_length

    tb_summary_writer = tb and tb.SummaryWriter(opt.checkpoint_path)

    infos = {}
    histories = {}
    if opt.start_from is not None:
        # open old infos and check if models are compatible
        with open(os.path.join(opt.checkpoint_path, 'infos_' + opt.id + format(int(opt.start_from),'04') + '.pkl')) as f:
            infos = cPickle.load(f)
            saved_model_opt = infos['opt']
            need_be_same=["caption_model", "rnn_type", "rnn_size", "num_layers"]
            for checkme in need_be_same:
                assert vars(saved_model_opt)[checkme] == vars(opt)[checkme], "Command line argument and saved model disagree on '%s' " % checkme

        if os.path.isfile(os.path.join(opt.checkpoint_path, 'histories_' + opt.id + format(int(opt.start_from),'04') + '.pkl')):
            with open(os.path.join(opt.checkpoint_path, 'histories_' + opt.id + format(int(opt.start_from),'04') + '.pkl')) as f:
                histories = cPickle.load(f)

    iteration = infos.get('iter', 0)
    epoch = infos.get('epoch', 0)

    val_result_history = histories.get('val_result_history', {})
    loss_history = histories.get('loss_history', {})
    lr_history = histories.get('lr_history', {})
    ss_prob_history = histories.get('ss_prob_history', {})

    loader.iterators = infos.get('iterators', loader.iterators)
    loader.split_ix = infos.get('split_ix', loader.split_ix)
    if opt.load_best_score == 1:
        best_val_score = infos.get('best_val_score', None)

    model = models.setup(opt).cuda()

    dp_model = model

    update_lr_flag = True
    # Assure in training mode
    dp_model.train()

    crit = utils.LanguageModelCriterion()
    rl_crit = utils.RewardCriterion()
    CE_ac = utils.CE_ac()

    optim_para = model.parameters()
    optimizer = utils.build_optimizer(optim_para, opt)
    optimizer_mem = optim.Adam([model.memory_cell], opt.learning_rate, (opt.optim_alpha, opt.optim_beta),
                               opt.optim_epsilon,
                               weight_decay=opt.weight_decay)

    # Load the optimizer
    if vars(opt).get('start_from', None) is not None and os.path.isfile(
            os.path.join(opt.checkpoint_path, 'optimizer' + opt.id + format(int(opt.start_from),'04')+'.pth')):
        optimizer.load_state_dict(torch.load(os.path.join(
            opt.checkpoint_path, 'optimizer' + opt.id + format(int(opt.start_from),'04')+'.pth')))
        if os.path.isfile(
            os.path.join(opt.checkpoint_path, 'optimizer_mem' + opt.id + format(int(opt.start_from),'04')+'.pth')):
                optimizer_mem.load_state_dict(torch.load(os.path.join(
                    opt.checkpoint_path, 'optimizer_mem' + opt.id + format(int(opt.start_from), '04') + '.pth')))

    optimizer.zero_grad()
    optimizer_mem.zero_grad()
    accumulate_iter = 0
    train_loss = 0
    reward = np.zeros([1,1])
    sim_lambda = opt.sim_lambda

    while True:
        if update_lr_flag:
                # Assign the learning rate
            if epoch > opt.learning_rate_decay_start and opt.learning_rate_decay_start >= 0:
                frac = (epoch - opt.learning_rate_decay_start) // opt.learning_rate_decay_every
                decay_factor = opt.learning_rate_decay_rate  ** frac
                opt.current_lr = opt.learning_rate * decay_factor
            else:
                opt.current_lr = opt.learning_rate
            utils.set_lr(optimizer, opt.current_lr)
            # Assign the scheduled sampling prob
            if epoch > opt.scheduled_sampling_start and opt.scheduled_sampling_start >= 0:
                frac = (epoch - opt.scheduled_sampling_start) // opt.scheduled_sampling_increase_every
                opt.ss_prob = min(opt.scheduled_sampling_increase_prob  * frac, opt.scheduled_sampling_max_prob)
                model.ss_prob = opt.ss_prob

            # If start self critical training
            if opt.self_critical_after != -1 and epoch >= opt.self_critical_after:
                sc_flag = True
                init_scorer(opt.cached_tokens)
            else:
                sc_flag = False

            update_lr_flag = False
                
        start = time.time()
        # Load data from train split (0)
        data = loader.get_batch(opt.train_split)
        print('Read data:', time.time() - start)

        torch.cuda.synchronize()
        start = time.time()

        tmp = [data['labels'], data['masks'], data['mods']]
        tmp = [_ if _ is None else torch.from_numpy(_).cuda() for _ in tmp]
        labels, masks, mods = tmp

        tmp = [data['att_feats'], data['att_masks'], data['attr_feats'], data['attr_masks'],data['rela_feats'], data['rela_masks']]
        tmp = [_ if _ is None else torch.from_numpy(_).cuda() for _ in tmp]
        att_feats, att_masks, attr_feats, attr_masks, rela_feats, rela_masks = tmp

        rs_data = {}
        rs_data['att_feats'] = att_feats
        rs_data['att_masks'] = att_masks
        rs_data['attr_feats'] = attr_feats
        rs_data['attr_masks'] = attr_masks
        rs_data['rela_feats'] = rela_feats
        rs_data['rela_masks'] = rela_masks

        if not sc_flag:
            logits, cw_logits = dp_model(rs_data, labels)
            ac = CE_ac(logits,labels[:,1:], masks[:,1:])
            print('ac :{0}'.format(ac))
            loss_lan = crit(logits,labels[:,1:], masks[:,1:])
        else:
            gen_result, sample_logprobs, cw_logits = dp_model(rs_data,
                                                   opt={'sample_max':0}, mode='sample')
            reward = get_self_critical_reward(dp_model, rs_data, data, gen_result, opt)
            loss_lan = rl_crit(sample_logprobs, gen_result.data, torch.from_numpy(reward).float().cuda())

        loss_cw = crit(cw_logits, mods[:, 1:], masks[:, 1:])
        ac2 = CE_ac(cw_logits, mods[:, 1:], masks[:, 1:])
        print('ac :{0}'.format(ac2))
        if epoch < opt.step2_train_after:
            loss = loss_lan + sim_lambda*loss_cw
        else:
            loss = loss_lan

        accumulate_iter =  accumulate_iter + 1
        loss = loss/opt.accumulate_number
        loss.backward()
        if accumulate_iter % opt.accumulate_number == 0:
            utils.clip_gradient(optimizer, opt.grad_clip)
            optimizer.step()
            optimizer.zero_grad()

            utils.clip_gradient(optimizer_mem, opt.grad_clip)
            optimizer_mem.step()
            optimizer_mem.zero_grad()

            iteration += 1
            accumulate_iter = 0
            train_loss = loss.item()*opt.accumulate_number
            train_loss_lan = loss_lan.item()
            train_loss_cw = loss_cw.item()
            end = time.time()

            if not sc_flag:
                print("iter {} (epoch {}), train_loss = {:.3f}, time/batch = {:.3f}" \
                      .format(iteration, epoch, train_loss, end - start))
                print("train_loss_lan = {:.3f}, train_loss_cw = {:.3f}" \
                      .format(train_loss_lan, train_loss_cw))
            else:
                print("iter {} (epoch {}), avg_reward = {:.3f}, time/batch = {:.3f}" \
                      .format(iteration, epoch, np.mean(reward[:, 0]), end - start))
                print("train_loss_lan = {:.3f}, train_loss_cw = {:.3f}" \
                      .format(train_loss_lan, train_loss_cw))
            print('lr:{0}'.format(opt.current_lr))

        torch.cuda.synchronize()

        # Update the iteration and epoch
        if data['bounds']['wrapped']:
            epoch += 1
            update_lr_flag = True

        # Write the training loss summary
        if (iteration % opt.losses_log_every == 0) and (accumulate_iter % opt.accumulate_number == 0):
            add_summary_value(tb_summary_writer, 'train_loss', train_loss, iteration)
            add_summary_value(tb_summary_writer, 'train_loss_lan', train_loss_lan, iteration)
            add_summary_value(tb_summary_writer, 'train_loss_cw', train_loss_cw, iteration)
            add_summary_value(tb_summary_writer, 'learning_rate', opt.current_lr, iteration)
            add_summary_value(tb_summary_writer, 'scheduled_sampling_prob', model.ss_prob, iteration)
            add_summary_value(tb_summary_writer, 'ac', ac, iteration)
            if sc_flag:
                add_summary_value(tb_summary_writer, 'avg_reward', np.mean(reward[:,0]), iteration)

            loss_history[iteration] = train_loss if not sc_flag else np.mean(reward[:,0])
            lr_history[iteration] = opt.current_lr
            ss_prob_history[iteration] = model.ss_prob

        # make evaluation on validation set, and save model
        if (iteration % opt.save_checkpoint_every == 0) and (accumulate_iter % opt.accumulate_number == 0):
            # eval model
            eval_kwargs = {'split': 'test',
                               'dataset': opt.input_json}
            eval_kwargs.update(vars(opt))
            current_score=0

            best_flag = False
            if True: # if true
                save_id = iteration/opt.save_checkpoint_every
                if best_val_score is None or current_score > best_val_score:
                    best_val_score = current_score
                    best_flag = True
                checkpoint_path = os.path.join(opt.checkpoint_path, 'model'+opt.id+format(int(save_id),'04')+'.pth')
                torch.save(model.state_dict(), checkpoint_path)
                print("model saved to {}".format(checkpoint_path))
                optimizer_path = os.path.join(opt.checkpoint_path, 'optimizer'+opt.id+format(int(save_id),'04')+'.pth')
                torch.save(optimizer.state_dict(), optimizer_path)

                optimizer_mem_path = os.path.join(opt.checkpoint_path,
                                                  'optimizer_mem' + opt.id + format(int(save_id), '04') + '.pth')
                torch.save(optimizer_mem.state_dict(), optimizer_mem_path)

                memory_cell = dp_model.memory_cell.data.cpu().numpy()
                memory_cell_path = os.path.join(opt.checkpoint_path,
                                                'memory_cell' + opt.id + format(int(save_id), '04') + '.npz')
                np.savez(memory_cell_path, memory_cell=memory_cell)

                # Dump miscalleous informations
                infos['iter'] = iteration
                infos['epoch'] = epoch
                infos['iterators'] = loader.iterators
                infos['split_ix'] = loader.split_ix
                infos['best_val_score'] = best_val_score
                infos['opt'] = opt
                infos['vocab'] = loader.get_vocab()

                histories['val_result_history'] = val_result_history
                histories['loss_history'] = loss_history
                histories['lr_history'] = lr_history
                histories['ss_prob_history'] = ss_prob_history
                with open(os.path.join(opt.checkpoint_path, 'infos_'+opt.id+format(int(save_id),'04')+'.pkl'), 'wb') as f:
                    cPickle.dump(infos, f)
                with open(os.path.join(opt.checkpoint_path, 'histories_'+opt.id+format(int(save_id),'04')+'.pkl'), 'wb') as f:
                    cPickle.dump(histories, f)

                if best_flag:
                    checkpoint_path = os.path.join(opt.checkpoint_path, 'model-best.pth')
                    torch.save(model.state_dict(), checkpoint_path)
                    print("model saved to {}".format(checkpoint_path))
                    with open(os.path.join(opt.checkpoint_path, 'infos_'+opt.id+'-best.pkl'), 'wb') as f:
                        cPickle.dump(infos, f)

        # Stop if reaching max epochs
        if epoch >= opt.max_epochs and opt.max_epochs != -1:
            break

opt = opts.parse_opt()
os.environ["CUDA_VISIBLE_DEVICES"]=str(opt.gpu)
# memory_pool = torch.FloatTensor(15000, 3, 400, 200).cuda()
# del memory_pool
train(opt)
