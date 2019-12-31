from __future__ import print_function
import argparse
from timeit import default_timer as timer
import os
import pathlib
import pprint
import sys

import torch
import torch.optim as optim
from torch.backends import cudnn
from torch.autograd import Variable
import torchvision.utils as vutils
from tqdm import tqdm
import torch.autograd.profiler

PROJECT_DIR = pathlib.Path(__file__).absolute().parent.parent # main directory, the parent of src
if str(PROJECT_DIR) not in sys.path:
    sys.path.append(str(PROJECT_DIR))

import src.utils as utils


global loss_func
global avg_training_time_per_epoch
global avg_testing_time_per_epoch
global best_acc
global best_acc_epoch
best_acc = 0
best_acc_epoch = 0
avg_training_time_per_epoch = 0
avg_testing_time_per_epoch = 0

# print(PROJECT_DIR)
def train(model, data_loader, optimizer, epoch, train_mloss, train_rloss, train_acc, learning_rate, lr_wr, output_tensor):
    """
    Train CapsuleNet model on training set

    Args:
        model: The CapsuleNet model.
        data_loader: An interator over the dataset. It combines a dataset and a sampler.
        optimizer: Optimization algorithm.
        epoch: Current epoch.
    """
    print('===> Training mode')

    num_batches = len(data_loader) # iteration per epoch. e.g: 469
    total_step = args.epochs * num_batches
    epoch_tot_acc = 0

    # Switch to train mode
    model.train()

    if args.cuda:
        # When we wrap a Module in DataParallel for multi-GPUs
        model = model.module

    start_time = timer()

    for batch_idx, (data, target) in enumerate(tqdm(data_loader, unit='batch')):
        batch_size = data.size(0)
        global_step = batch_idx + (epoch * num_batches) - num_batches

        labels = target
        target_one_hot = utils.one_hot_encode(target, length=args.num_classes)
        assert target_one_hot.size() == torch.Size([batch_size, 10])

        data, target = Variable(data), Variable(target_one_hot)

        if args.cuda:
            data = data.to(args.device)
            target = target.to(args.device)
            labels = labels.to(args.device)

        # Train step - forward, backward and optimize
        optimizer.zero_grad()
        #utils.exponential_decay_LRR(optimizer, args.lr, global_step, args.decay_steps, args.decay_rate, args.staircase)
        # learning rate policies
        if args.find_lr:
            utils.find_lr(optimizer, global_step)

        elif args.exp_decay_lr:
            utils.exponential_decay_LRR(
                optimizer, args.lr, global_step, args.decay_steps, args.decay_rate, args.staircase)

        elif args.one_cycle_policy:
            utils.one_cycle_policy(optimizer, args.lr, global_step, total_step)

        elif args.warm_restarts:
            # lr_wr.update_lr(optimizer, num_batches)
            lr_wr.update_lr(optimizer)

        output, reconstruction = model(data, labels, True)
        # utils.write_tensor(output, output_tensor)
        loss, margin_loss, recon_loss = loss_func(
            output, target, args.regularization_scale, reconstruction, data, args.device, batch_size)
        loss.backward()
        optimizer.step()

        for param_group in optimizer.param_groups:
            lr_temp = param_group['lr']
        learning_rate.write('%.10f \n' % lr_temp)

        # Calculate accuracy for each step and average accuracy for each epoch
        acc = utils.accuracy(output, labels, args.cuda)
        epoch_tot_acc += acc
        epoch_avg_acc = epoch_tot_acc / (batch_idx + 1)

        train_mloss.write('%.6f \n' % margin_loss)
        train_rloss.write('%.6f \n' % recon_loss)
        train_acc.write('%.6f \n' % acc)

        # Print losses
        if batch_idx % args.log_interval == 0:
            template = 'Epoch {}/{}, ' \
                    'Step {}/{}: ' \
                    '[Total loss: {:.6f},' \
                    '\tMargin loss: {:.6f},' \
                    '\tReconstruction loss: {:.6f},' \
                    '\tBatch accuracy: {:.6f},' \
                    '\tAccuracy: {:.6f}]'
            tqdm.write(template.format(
                epoch,
                args.epochs,
                global_step,
                total_step,
                loss.data.item(),
                margin_loss.data.item(),
                recon_loss.data.item() if args.use_reconstruction_loss else 0,
                acc,
                epoch_avg_acc))

    # Print time elapsed for an epoch
    end_time = timer()

    global avg_training_time_per_epoch

    avg_training_time_per_epoch = (avg_training_time_per_epoch * (epoch - 1) + end_time - start_time) / epoch

    print('Time elapsed for epoch {}: {:.0f}s.'.format(epoch, end_time - start_time))


def test(model, data_loader, num_train_batches, epoch, test_mloss, test_rloss, test_acc, directory):
    """
    Evaluate model on validation set

    Args:
        model: The CapsuleNet model.
        data_loader: An interator over the dataset. It combines a dataset and a sampler.
    """
    print('===> Evaluate mode')

    # Switch to evaluate mode
    model.eval()

    if args.cuda:
        # When we wrap a Module in DataParallel for multi-GPUs
        model = model.module

    loss = 0
    margin_loss = 0
    recon_loss = 0

    correct = 0

    num_batches = len(data_loader)

    global_step = epoch * num_train_batches + num_train_batches

    start_time = timer()

    for data, target in data_loader:
        with torch.no_grad():
            batch_size = data.size(0)
            target_indices = target
            target_one_hot = utils.one_hot_encode(target_indices, length=args.num_classes)
            assert target_one_hot.size() == torch.Size([batch_size, 10])

            target = target_one_hot

            if args.cuda:
                data = data.to(args.device)
                target = target.to(args.device)
                target_indices.to(args.device)

            # Output predictions
            output, reconstruction = model(data, target_indices, False) # output from DigitCaps (out_digit_caps)

            # Sum up batch loss
            t_loss, m_loss, r_loss = loss_func(
                output, target, args.regularization_scale, reconstruction, data, args.device, batch_size)
            loss += t_loss.data
            margin_loss += m_loss.data
            recon_loss += r_loss.data

            # Count number of correct predictions
            # v_magnitude shape: [128, 10, 1, 1]
            v_magnitude = torch.sqrt((output**2).sum(dim=2, keepdim=True))
            # pred shape: [128, 1, 1, 1]
            pred = v_magnitude.data.max(1, keepdim=True)[1].cpu()
            correct += pred.eq(target_indices.view_as(pred)).sum()


    # Get the reconstructed images of the last batch
    if args.use_reconstruction_loss:
        reconstruction = model.decoder(output, target_indices, False)
        # Input image size and number of channel.
        # By default, for MNIST, the image width and height is 28x28 and 1 channel for black/white.
        image_width = args.input_width
        image_height = args.input_height
        image_channel = args.num_conv_in_channels
        recon_img = reconstruction.view(-1, image_channel, image_width, image_height)
        assert recon_img.size() == torch.Size([batch_size, image_channel, image_width, image_height])

        # Save the image into file system
        utils.save_image(recon_img, directory / ('recons_image_test_{}_{}.png'.format(epoch, global_step)))
        utils.save_image(data, directory /
                         ('original_image_test_{}_{}.png'.format(epoch, global_step)))

    end_time = timer()

    # Log test losses
    loss /= num_batches
    margin_loss /= num_batches
    recon_loss /= num_batches

    # Log test accuracies
    num_test_data = len(data_loader.dataset)
    accuracy = correct / num_test_data
    accuracy_percentage = float(correct) * 100.0 / float(num_test_data)

    test_mloss.write('%.6f \n' % margin_loss)
    test_rloss.write('%.6f \n' % recon_loss)
    test_acc.write('%.4f \n' % accuracy_percentage)

    # Print test losses and accuracy
    print('Test: [Loss: {:.6f},' \
        '\tMargin loss: {:.6f},' \
        '\tReconstruction loss: {:.6f}]'.format(
            loss,
            margin_loss,
            recon_loss if args.use_reconstruction_loss else 0))
    print('Test Accuracy: {}/{} ({:.2f}%)\n'.format(
        correct, num_test_data, accuracy_percentage))


    global avg_testing_time_per_epoch
    avg_testing_time_per_epoch = (
        avg_testing_time_per_epoch * (epoch - 1) + end_time - start_time) / epoch

    global best_acc
    global best_acc_epoch
    if accuracy_percentage > best_acc:
        best_acc = accuracy_percentage
        best_acc_epoch = epoch
        test_loader = data_loader
        utils.dump(utils.make_full_checkpoint_obj(locals(), globals()), directory / 'trained_model/FP32_model')


def main(arguments=None):
    """The main function
    Entry point.
    """
    global loss_func
    global best_acc
    best_acc = 0
    global args

    # Setting the hyper parameters
    parser = argparse.ArgumentParser(description='Example of Capsule Network')
    parser.add_argument('--epochs', type=int, default=10,
                        help='number of training epochs. default=10')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='learning rate. default=0.01')
    parser.add_argument('--batch-size', type=int, default=128,
                        help='training batch size. default=128')
    parser.add_argument('--test-batch-size', type=int,
                        default=128, help='testing batch size. default=128')
    parser.add_argument('--log-interval', type=int, default=10,
                        help='how many batches to wait before logging training status. default=10')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training. default=false')
    parser.add_argument('--device', type=str, default='cuda:0',
                        help='select the gpu.  default=cuda:0')
    parser.add_argument('--threads', type=int, default=4,
                        help='number of threads for data loader to use. default=4')
    parser.add_argument('--seed', type=int, default=42,
                        help='random seed for training. default=42')
    parser.add_argument('--num_conv_in_channels', type=int, default=1,
                        help='number of channels in input to first Conv Layer.  default=1')
    parser.add_argument('--num_conv_out_channels', type=int, default=256,
                        help='number of channels in output from first Conv Layer.  default=256')
    parser.add_argument('--conv-kernel', type=int, default=9,
                        help='kernel size of Conv Layer.  default=9')
    parser.add_argument('--conv-stride', type=int, default=1,
                        help='stride of first Conv Layer.  default=1')
    parser.add_argument('--num-primary-channels', type=int, default=32,
                        help='channels produced by PimaryCaps layer.  default=32')
    parser.add_argument('--primary-caps-dim', type=int, default=8,
                        help='dimension of capsules in PrimaryCaps layer.  default=8')
    parser.add_argument('--primary-kernel', type=int, default=9,
                        help='kernel dimension for PrimaryCaps layer.  default=9')
    parser.add_argument('--primary-stride', type=int, default=2,
                        help='stride for PrimaryCaps layer.  default=2')
    parser.add_argument('--num-classes', type=int, default=10,
                        help='number of output classes.  default=10 for MNIST')
    parser.add_argument('--digit-caps-dim', type=int, default=16,
                        help='dimension of capsules in DigitCaps layer. default=16')
    parser.add_argument('--dec1-dim', type=int, default=512,
                        help='output dimension of first layer in decoder.  default=512')
    parser.add_argument('--dec2-dim', type=int, default=1024,
                        help='output dimension of seconda layer in decoder.  default=1024')
    parser.add_argument('--num-routing', type=int,
                        default=3, help='number of routing iteration. default=3')
    parser.add_argument('--use-reconstruction-loss', type=utils.str2bool, nargs='?', default=True,
                        help='use an additional reconstruction loss. default=True')
    parser.add_argument('--regularization-scale', type=float, default=0.0005,
                        help='regularization coefficient for reconstruction loss. default=0.0005')
    parser.add_argument('--dataset', help='the name of dataset (mnist, cifar10)', default='mnist')
    parser.add_argument('--input-width', type=int,
                        default=28, help='input image width to the convolution. default=28 for MNIST')
    parser.add_argument('--input-height', type=int,
                        default=28, help='input image height to the convolution. default=28 for MNIST')
    parser.add_argument('--directory', type=str, default=PROJECT_DIR / 'results',
                        help='directory to store results')
    parser.add_argument('--data-directory', type=str, default=PROJECT_DIR / 'data',
                        help='directory to store data')
    parser.add_argument('--description', type=str, default='no description',
                        help='description to store together with results')
    parser.add_argument('--exp-decay-lr', action='store_true', default=False,
                        help='use exponential decay of learning rate')
    parser.add_argument('--decay-steps', type=int, default=4000,
                        help='decay steps for exponential learning rate adjustment.  default = 2000')
    parser.add_argument('--decay-rate', type=float, default=0.96,
                        help='decay rate for exponential learning rate adjustment.  default=1 (no adjustment)')
    parser.add_argument('--staircase', action='store_true', default=False,
                        help='activate staircase for learning rate adjustment')
    # one cycle policy
    parser.add_argument('--one-cycle-policy', action='store_true', default=False,
                        help='use one cycle policy for learning rate')
    # warm restarts
    parser.add_argument('--warm-restarts', action='store_true', default=False,
                        help='use warm restarts of the learning rate')
    parser.add_argument('--Ti', type=float, default=10.0,
                        help='number of epochs of a cycle of the warm restarts')
    parser.add_argument('--Tmult', type=float, default=1.0,
                        help='multiplier factor for the warm restarts')
    # adaptive batch size
    parser.add_argument('--adabatch', action='store_true', default=False,
                        help='activate adabatch.  default False')
    parser.add_argument('--adapow', type=int,
                        default=2, help='power of two for adabatch size')
    # weight sharing
    parser.add_argument('--conv-shared-weights', type=int, default=0)
    parser.add_argument('--primary-shared-weights', type=int, default=0)
    parser.add_argument('--digit-shared-weights', type=int, default=0)
    parser.add_argument('--conv-shared-bias', type=int, default=0)
    # small decoder
    parser.add_argument('--small-decoder', action='store_true', default=False,
                        help='enables the small decoder instead of the standard one')
    # restart option
    parser.add_argument('--restart-training', action='store_true', default=False)
    # squash approx
    parser.add_argument('--squash-approx', action='store_true', default=False)

    # find best learning rate interval
    parser.add_argument('--find-lr', action='store_true', default=False,
                        help='train to find the best learning rate')

    # normalize or not the inputs to the net (not normalized is better)
    parser.add_argument('--normalize-input', action='store_true',
                        default=False,
                        help='enables normalization and disables random '
                        'cropping the '
                        'inputs with padding 2')

    # use new / old version of the model
    parser.add_argument('--old-model', action='store_true',
                        default=False,
                        help='uses old model')

    args = parser.parse_args(args=arguments)

    args.directory = pathlib.Path(args.directory)

    print(args)

    if args.old_model:
        from src.model.model import Net
        import src.model.functions as func
        ModelToUse = Net
        def loss_func(output, target, regularization_scale, reconstruction, data, device, batch_size):
            return func.loss(output, reconstruction, target, data, regularization_scale, device)
    else:
        from src.model.layers import CapsNet
        from src.model.layers import loss_func as loss_func_internal
        ModelToUse = CapsNet
        def loss_func(output, target, regularization_scale, reconstruction, data, device, batch_size):
            return loss_func_internal(output, target, regularization_scale, reconstruction, data.view(batch_size, -1), device)


    # Check GPU or CUDA is available
    args.cuda = not args.no_cuda and torch.cuda.is_available()
    if not args.cuda:
        args.device = 'cpu'

    # Get reproducible results by manually seed the random number generator
    torch.manual_seed(args.seed)
    if args.cuda:
        torch.cuda.manual_seed(args.seed)

    # Load data
    train_loader, test_loader = utils.load_data(args)

    if args.adabatch:
        temp_bs = args.batch_size

        args.batch_size = 2**(args.adapow)
        train_loader1, _ = utils.load_data(args)

        args.batch_size = 2**(args.adapow)
        train_loader2, _ = utils.load_data(args)

        args.batch_size = 2**(args.adapow)
        train_loader3, _ = utils.load_data(args)

        args.batch_size = temp_bs

    # Build Capsule Network
    print('===> Building model')
    model = ModelToUse(input_wh = args.input_width,
                num_conv_in_channels = args.num_conv_in_channels,
                num_conv_out_channels = args.num_conv_out_channels,
                conv_kernel = args.conv_kernel,
                conv_stride = args.conv_stride,
                num_primary_channels = args.num_primary_channels,
                primary_caps_dim = args.primary_caps_dim,
                primary_kernel = args.primary_kernel,
                primary_stride = args.primary_stride,
                num_classes = args.num_classes,
                digit_caps_dim = args.digit_caps_dim,
                iter = args.num_routing,
                dec1_dim = args.dec1_dim,
                dec2_dim = args.dec2_dim,
                cuda_enabled = args.cuda,
                device = args.device,
                regularization_scale = args.regularization_scale,
                conv_shared_weights=args.conv_shared_weights,
                primary_shared_weights=args.primary_shared_weights,
                digit_shared_weights=args.digit_shared_weights,
                conv_shared_bias=args.conv_shared_bias,
                small_decoder=args.small_decoder,
                squash_approx=args.squash_approx)
    # Optimizer
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    # optimizer = optim.SGD(model.parameters(), lr=args.lr)
    lr_wr = utils.custom_warm_restarts(args.lr, args.lr*10)
    starting_epoch = 1

    if args.cuda:
        print('Utilize GPUs for computation')
        print('Number of GPU available', torch.cuda.device_count())
        model.to(args.device)
        cudnn.benchmark = True
        model = torch.nn.DataParallel(model)

    args.file_flag = 'w'
    if args.restart_training:
        args.file_flag = args.file_flag
        p = pathlib.Path(args.directory) / + 'trained_model'
        if p.exists():
            l = sorted(list(p.iterdir()))
            if l:
                f = l[-1]
                pckl = utils.load(str(f))
                model.load_state_dict(pckl['model_state_dict'])
                optimizer.load_state_dict(pckl['optimizer_state_dict'])
                lr_wr.__dict__ = pckl['lr_wr']
                starting_epoch = pckl['epoch']

    # Print the model architecture and parameters
    print('Model architectures:\n{}\n'.format(model))

    print('Parameters and size:')
    for name, param in model.named_parameters():
        print('{}: {}'.format(name, list(param.size())))

    # CapsNet has:
    # - 8.2M parameters and 6.8M parameters without the reconstruction subnet on MNIST.
    # - 11.8M parameters and 8.0M parameters without the reconstruction subnet on CIFAR10.
    num_params = sum([param.nelement() for param in model.parameters()])

    # The coupling coefficients c_ij are not included in the parameter list,
    # we need to add them manually, which is 1152 * 10 = 11520 (on MNIST) or 2048 * 10 (on CIFAR10)
    print('\nTotal number of parameters: {}\n'.format(num_params + (11520 if args.dataset in ('mnist', 'fashionmnist') else 20480)))

    # Make model checkpoint directory
    if not (args.directory / 'trained_model').is_dir():
        (args.directory / 'trained_model').mkdir(parents=True, exist_ok=True)

    # files to store accuracies and losses
    train_mloss = args.directory / 'train_margin_loss.txt'
    train_rloss = args.directory / 'train_reconstruction_loss.txt'
    train_acc = args.directory / 'train_accuracy.txt'

    test_mloss = args.directory / 'test_margin_loss.txt'
    test_rloss = args.directory / 'test_reconstruction_loss.txt'
    test_acc = args.directory / 'test_accuracy.txt'

    learning_rate = args.directory / 'learning_rate.txt'
    output_tensor = args.directory / 'output_tensor.txt'

    n_parameters = args.directory / 'n_parameters.txt'
    with open(n_parameters, args.file_flag) as f:
        f.write('{}\n'.format(num_params + (11520 if args.dataset == 'mnist' else 20480)))

    arguments_file = args.directory / 'arguments.txt'
    with open(arguments_file, args.file_flag) as f:
        pprint.pprint(args.__dict__, stream=f)

    description = args.directory / 'details.txt'
    description = open(description, args.file_flag)
    description.write(args.description)
    description.close()

    train_mloss = open(train_mloss, args.file_flag)
    train_rloss = open(train_rloss, args.file_flag)
    train_acc = open(train_acc, args.file_flag)
    test_mloss = open(test_mloss, args.file_flag)
    test_rloss = open(test_rloss, args.file_flag)
    test_acc = open(test_acc, args.file_flag)
    learning_rate = open(learning_rate, args.file_flag)
    output_tensor = open(output_tensor, args.file_flag)

    utils.dump(utils.make_dataset_obj(locals(), globals()), args.directory / 'trained_model' / 'dataset')

    # Train and test
    try:
        for epoch in range(starting_epoch, args.epochs + 1):

            if not args.adabatch:
                train(model, train_loader, optimizer, epoch, train_mloss, train_rloss, train_acc, learning_rate, lr_wr, output_tensor)
                test(model, test_loader, len(train_loader), epoch, test_mloss, test_rloss, test_acc, args.directory)
            else:
                if (1 <= epoch <= 3):
                    train(model, train_loader, optimizer, epoch, train_mloss, train_rloss,
                        train_acc, learning_rate, lr_wr, output_tensor)
                    test(model, test_loader, len(train_loader), epoch,
                        test_mloss, test_rloss, test_acc, args.directory)
                elif (4 <= epoch <= 33):
                    args.batch_size = 2**(args.adapow)
                    train(model, train_loader1, optimizer, epoch, train_mloss,
                        train_rloss, train_acc, learning_rate, lr_wr, output_tensor)
                    test(model, test_loader, len(train_loader), epoch,
                        test_mloss, test_rloss, test_acc, args.directory)
                elif (34 <= epoch <= 63):
                    args.batch_size = 2**(args.adapow+1)
                    train(model, train_loader2, optimizer, epoch, train_mloss,
                        train_rloss, train_acc, learning_rate, lr_wr, output_tensor)
                    test(model, test_loader, len(train_loader), epoch,
                        test_mloss, test_rloss, test_acc, args.directory)
                else:
                    args.batch_size = 2**(args.adapow+2)
                    train(model, train_loader3, optimizer, epoch, train_mloss,
                        train_rloss, train_acc, learning_rate, lr_wr, output_tensor)
                    test(model, test_loader, len(train_loader), epoch,
                        test_mloss, test_rloss, test_acc, args.directory)
            train_mloss.flush()
            train_rloss.flush()
            train_acc.flush()
            test_mloss.flush()
            test_rloss.flush()
            test_acc.flush()
            learning_rate.flush()
            output_tensor.flush()

            # Save model checkpoint
            utils.checkpoint(utils.make_partial_checkpoint_obj(locals(), globals()), epoch, directory=args.directory)
    except KeyboardInterrupt:
        print("\n\n\nKeyboardInterrupt, Interrupting...")

    train_mloss.close()
    train_rloss.close()
    train_acc.close()
    test_mloss.close()
    test_rloss.close()
    test_acc.close()
    learning_rate.close()
    output_tensor.close()
    with open(args.directory / 'best_accuracy.txt', args.file_flag) as f:
        f.write("%.10f,%d\n" % (best_acc, best_acc_epoch))
    print('\n\nBest Accuracy: ' + str(best_acc) + '%%\nReached at epoch: %d\n\n' % best_acc_epoch)

    global avg_training_time_per_epoch
    global avg_testing_time_per_epoch

    with open(args.directory / 'average_training_time_per_epoch.txt', args.file_flag) as f:
        f.write("%.10f\n" % avg_training_time_per_epoch)
    print('Average time per training epoch: %.10f\n\n' % avg_training_time_per_epoch)
    with open(args.directory / 'average_testing_time_per_epoch.txt', args.file_flag) as f:
        f.write("%.10f\n" % avg_testing_time_per_epoch)
    print('Average time per testing epoch: %.10f\n\n' % avg_testing_time_per_epoch)

if __name__ == "__main__":
    main()
