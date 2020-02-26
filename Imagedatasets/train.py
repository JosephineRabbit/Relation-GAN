#coding=utf-8
import time
from options.train_options import TrainOptions
from data import CreateDataLoader
from models import create_model
from util.logger import Logger


def print_current_losses(epoch, i, losses, t, t_data):
    message = '(epoch: %d, iters: %d, time: %.3f, data: %.3f) ' % (epoch, i, t, t_data)
    for k, v in losses.items():
        message += '%s: %.3f ' % (k, v)
    print(message)

if __name__ == '__main__':

    opt = TrainOptions().parse()
    tensorbord_log = Logger('../log_all/' + opt.name)
    data_loader = CreateDataLoader(opt)
    dataset = data_loader.load_data()
    dataset_size = len(data_loader)
    print('#training images = %d' % dataset_size)

    model = create_model(opt)
    if opt.which_step != 'latest':
        total_steps = int(opt.which_step)*opt.save_step_freq
    else:
        total_steps = 0
    count_print_loss = 0
    for epoch in range(opt.epoch_count, opt.niter + opt.niter_decay + 1,):
        epoch_start_time = time.time()
        iter_data_time = time.time()
        epoch_iter = 0
        for i, data in enumerate(dataset):
            iter_start_time = time.time()
            if total_steps % opt.print_freq == 0:
                t_data = iter_start_time - iter_data_time
            total_steps += 1
            epoch_iter += opt.batchSize
            model.set_input(data)

            model.optimize_parameters()
            # tensorbord
            if total_steps % opt.display_freq == 0:
                for tag, value in model.get_current_losses().items():
                    tensorbord_log.scalar_summary(tag, value, total_steps+1)
                for tag, images in model.get_current_visuals().items():
                    images = images.cpu()[0].unsqueeze(0)
                    tensorbord_log.image_summary(tag,images, total_steps + 1)
                if total_steps % (opt.display_freq*2) == 0:
                    net_all = model.get_all_model()
                    for net_name in net_all:
                        for tag, value in net_all[net_name].named_parameters():
                            tag = net_name+'_'+tag
                            tag = tag.replace('.', '/')
                            tensorbord_log.histo_summary(tag, value.data.cpu().numpy(), total_steps + 1)

            if total_steps % opt.print_freq == 0:
                losses = model.get_current_losses()
                t = (time.time() - iter_start_time) / opt.batchSize
                print_current_losses(epoch, epoch_iter, losses, t, t_data)

            if total_steps % opt.save_latest_freq == 0:
                print('saving the latest model (epoch %d, total_steps %d)' %
                      (epoch, total_steps))
                model.save_networks('latest')
            if total_steps % opt.save_step_freq == 0:
                model.save_networks(total_steps//opt.save_step_freq)
            iter_data_time = time.time()
        model.save_networks('latest')

        print('End of epoch %d / %d \t Time Taken: %d sec' %
              (epoch, opt.niter + opt.niter_decay, time.time() - epoch_start_time))
        # model.update_learning_rate()
    model.save_networks('latest')