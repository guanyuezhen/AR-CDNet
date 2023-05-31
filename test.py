import sys

sys.path.insert(0, '.')
import torch
import scipy.io as scio
import torch.backends.cudnn as cudnn
from torch.nn.parallel import gather
import torch.optim.lr_scheduler
import datasets.dataset as myDataLoader
import datasets.Transforms as myTransforms
from utils.metric_tool import ConfuseMatrixMeter
from PIL import Image
import os, time
import numpy as np
from argparse import ArgumentParser
from models.model import get_model


@torch.no_grad()
def val(args, val_loader, model, vis_dir):
    model.eval()

    salEvalVal = ConfuseMatrixMeter(n_class=2)

    total_batches = len(val_loader)
    print(len(val_loader))

    for iter, batched_inputs in enumerate(val_loader):

        img, target = batched_inputs
        img_name = val_loader.sampler.data_source.file_list[iter]
        start_time = time.time()

        if args.onGPU == True:
            img = img.cuda()
            target = target.cuda()

        img_var = torch.autograd.Variable(img).float()
        target_var = torch.autograd.Variable(target).float()

        # run the model
        change_mask, mask_d2, mask_d3, mask_d4, mask_d5, boundary_mask = model(img_var)
        output = change_mask

        change_pre = change_mask.detach()
        uncertainty_gt = torch.mul(target_var, (1 - change_pre)) + torch.mul(change_pre, (1 - target_var))
        #
        # import utils.torchutils as vis
        # vis.visulize_features(uncertainty_gt)

        pred = torch.where(output > 0.5, torch.ones_like(output), torch.zeros_like(output)).long()

        # torch.cuda.synchronize()
        time_taken = time.time() - start_time

        # compute the confusion matrix
        if args.onGPU and torch.cuda.device_count() > 1:
            output = gather(pred, 0, dim=0)

        # save change maps
        pr = pred[0, 0].cpu().numpy()
        gt = target_var[0, 0].cpu().numpy()
        index_tp = np.where(np.logical_and(pr == 1, gt == 1))
        index_fp = np.where(np.logical_and(pr == 1, gt == 0))
        index_tn = np.where(np.logical_and(pr == 0, gt == 0))
        index_fn = np.where(np.logical_and(pr == 0, gt == 1))
        #
        map = np.zeros([gt.shape[0], gt.shape[1], 3])
        map[index_tp] = [255, 255, 255]  # white
        map[index_fp] = [255, 0, 0]  # red
        map[index_tn] = [0, 0, 0]  # black
        map[index_fn] = [0, 255, 255]  # Cyan

        change_map = Image.fromarray(np.array(map, dtype=np.uint8))
        change_map.save(vis_dir + img_name)

        import matplotlib as mpl
        color_map = mpl.cm.get_cmap('jet')

        boundary_mask = boundary_mask[0, 0].cpu().numpy()
        boundary_mask = color_map(boundary_mask) * 255
        boundary_mask = Image.fromarray(np.array(boundary_mask, dtype=np.uint8))
        boundary_mask.save(vis_dir + 'confidence_pre_' + img_name)

        uncertainty_gt = uncertainty_gt[0, 0].cpu().numpy()
        uncertainty_gt = color_map(uncertainty_gt) * 255
        uncertainty_gt = Image.fromarray(np.array(uncertainty_gt, dtype=np.uint8))
        uncertainty_gt.save(vis_dir + 'confidence_gt_' + img_name)

        change_mask = change_mask[0, 0].cpu().numpy() * 255
        change_mask = Image.fromarray(np.array(change_mask, dtype=np.uint8))
        change_mask.save(vis_dir + 'change_pre_' + img_name)

        f1 = salEvalVal.update_cm(pr, gt)

        if iter % 5 == 0:
            print('\r[%d/%d] F1: %3f time: %.3f' % (iter, total_batches, f1, time_taken),
                  end='')

    scores = salEvalVal.get_scores()

    return scores


def val_change_detection(args):

    torch.backends.cudnn.benchmark = True
    SEED = 2333
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)

    model = get_model()

    args.save_dir = args.save_dir + '_iter_' + str(args.max_steps) + '_lr_' + str(args.lr) + '/'

    args.vis_dir_1 = './Predict/LEVIR/'
    args.vis_dir_2 = './Predict/BCDD/'

    args.test_data_root_1 = '/mnt/2800c818-54bc-4e2a-83d3-f418982b79e6/Change Detection/Datasets_BCD/LEVIR+TR_VAL_TE'
    # args.test_data_root_1 = '/mnt/2800c818-54bc-4e2a-83d3-f418982b79e6/Change Detection/Methods_BCD/TDRNet/CDNet/samples'

    args.test_data_root_2 = '/mnt/2800c818-54bc-4e2a-83d3-f418982b79e6/Change Detection/Datasets_BCD/BCDD-512_TE'

    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)

    if not os.path.exists(args.vis_dir_1):
        os.makedirs(args.vis_dir_1)

    if not os.path.exists(args.vis_dir_2):
        os.makedirs(args.vis_dir_2)

    if args.onGPU:
        model = model.cuda()

    total_params = sum([np.prod(p.size()) for p in model.parameters()])
    print('Total network parameters (excluding idr): ' + str(total_params))

    mean = [0.406, 0.456, 0.485, 0.406, 0.456, 0.485]
    std = [0.225, 0.224, 0.229, 0.225, 0.224, 0.229]

    # compose the data with transforms
    valDataset = myTransforms.Compose([
        myTransforms.Normalize(mean=mean, std=std),
        myTransforms.Scale(args.inWidth, args.inHeight),
        myTransforms.ToTensor()
    ])

    test_data_1 = myDataLoader.Dataset("test", file_root=args.test_data_root_1, transform=valDataset)
    testLoader_1 = torch.utils.data.DataLoader(
        test_data_1, shuffle=False,
        batch_size=args.batch_size, num_workers=args.num_workers, pin_memory=False)

    test_data_2 = myDataLoader.Dataset("test", file_root=args.test_data_root_2, transform=valDataset)
    testLoader_2 = torch.utils.data.DataLoader(
        test_data_2, shuffle=False,
        batch_size=args.batch_size, num_workers=args.num_workers, pin_memory=False)

    if args.onGPU:
        cudnn.benchmark = True

    logFileLoc = args.save_dir + args.logFile
    if os.path.isfile(logFileLoc):
        logger = open(logFileLoc, 'a')
    else:
        logger = open(logFileLoc, 'w')
        logger.write("Parameters: %s" % (str(total_params)))
        logger.write(
            "\n%s\t%s\t%s\t%s\t%s\t%s" % ('Epoch', 'Kappa', 'IoU', 'F1', 'R', 'P'))
    logger.flush()

    # load the model
    model_file_name = args.save_dir + 'best_model.pth'
    state_dict = torch.load(model_file_name)
    model.load_state_dict(state_dict)

    score_test_LEVIR = val(args, testLoader_1, model, args.vis_dir_1)
    torch.cuda.empty_cache()
    print("\nLEVIR_Test :\t Kappa (te) = %.4f\t IoU (te) = %.4f\t F1 (te) = %.4f\t R (te) = %.4f\t P (te) = %.4f" \
          % (score_test_LEVIR['Kappa'], score_test_LEVIR['IoU'], score_test_LEVIR['F1'], score_test_LEVIR['recall'], score_test_LEVIR['precision']))
    logger.write("\n%s\t\t%.4f\t\t%.4f\t\t%.4f\t\t%.4f\t\t%.4f" % ('LEVIR_Test',
                                                                   score_test_LEVIR['Kappa'],
                                                                   score_test_LEVIR['IoU'],
                                                                   score_test_LEVIR['F1'],
                                                                   score_test_LEVIR['recall'],
                                                                   score_test_LEVIR['precision']))
    logger.flush()
    scio.savemat(args.vis_dir_1 + 'results.mat', score_test_LEVIR)

    score_test_BCDD = val(args, testLoader_2, model, args.vis_dir_2)
    torch.cuda.empty_cache()
    print("\nBCDD_Test :\t Kappa (te) = %.4f\t IoU (te) = %.4f\t F1 (te) = %.4f\t R (te) = %.4f\t P (te) = %.4f" \
          % (score_test_BCDD['Kappa'], score_test_BCDD['IoU'], score_test_BCDD['F1'], score_test_BCDD['recall'], score_test_BCDD['precision']))
    logger.write("\n%s\t\t%.4f\t\t%.4f\t\t%.4f\t\t%.4f\t\t%.4f" % ('BCDD_Test',
                                                                   score_test_BCDD['Kappa'],
                                                                   score_test_BCDD['IoU'],
                                                                   score_test_BCDD['F1'],
                                                                   score_test_BCDD['recall'],
                                                                   score_test_BCDD['precision']))
    logger.flush()
    scio.savemat(args.vis_dir_2 + 'results.mat', score_test_BCDD)

    logger.close()


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--file_root', default="LEVIR", help='Data directory')
    parser.add_argument('--inWidth', type=int, default=512, help='Width of RGB image')
    parser.add_argument('--inHeight', type=int, default=512, help='Height of RGB image')
    parser.add_argument('--max_steps', type=int, default=20000, help='Max. number of iterations')
    parser.add_argument('--num_workers', type=int, default=4, help='No. of parallel threads')
    parser.add_argument('--batch_size', type=int, default=1, help='Batch size')
    parser.add_argument('--lr', type=float, default=5e-4, help='Initial learning rate')
    parser.add_argument('--lr_mode', default='poly', help='Learning rate policy')
    parser.add_argument('--save_dir', default='./weights/model', help='Directory to save the results')
    parser.add_argument('--logFile', default='trainValLog.txt',
                        help='File that stores the training and validation logs')
    parser.add_argument('--onGPU', default=True, type=lambda x: (str(x).lower() == 'true'),
                        help='Run on CPU or GPU. If TRUE, then GPU.')
    parser.add_argument('--weight', default='', type=str, help='pretrained weight, can be a non-strict copy')
    parser.add_argument('--ms', type=int, default=0, help='apply multi-scale training, default False')

    args = parser.parse_args()
    print('Called with args:')
    print(args)

    val_change_detection(args)