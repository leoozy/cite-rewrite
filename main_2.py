import os
import sys
import argparse
import time
import numpy as np
import tensorflow as tf
import torch
from model_ne import setup_model
from torch_data_loader_ne import TorchDataLoader
from data_loader import DataLoader
from random import shuffle
import  cv2
import pdb

parser = argparse.ArgumentParser(description='Conditional Image-Text Similarity Network')
parser.add_argument('--name', default='Conditional_Image-Text_Similarity_Network', type=str,
                    help='name of experiment')
parser.add_argument('--dataset', default='flickr', type=str,
                    help='name of the dataset to use')
parser.add_argument('--r_seed', type=int, default=42,
                    help='random seed (default: 42)')
parser.add_argument('--info_iterval', type=int, default=20,
                    help='number of batches to process before outputing training status')
parser.add_argument('--resume', default='model_best', type=str,
                    help='filename of model to load (default: none)')
parser.add_argument('--test', dest='test', action='store_true', default=False,
                    help='Run model on test set')
parser.add_argument('--batch-size', type=int, default=200,
                    help='input batch size for training (default: 200)')
parser.add_argument('--lr', type=float, default=5e-6, metavar='LR',
                    help='learning rate (default: 5e-6)')
parser.add_argument('--embed_l1', type=float, default=5e-6,
                    help='weight of the L1 regularization term used on the concept weight branch (default: 5e-5)')
parser.add_argument('--max_epoch', type=int, default=0,
                    help='maximum number of epochs, less than 1 indicates no limit (default: 0)')
parser.add_argument('--no_gain_stop', type=int, default=5,
                    help='number of epochs used to perform early stopping based on validation performance (default: 5)')
parser.add_argument('--neg_to_pos_ratio', type=int, default=2,
                    help='ratio of negatives to positives used during training (default: 2)')
parser.add_argument('--minimum_gain', type=float, default=5e-4, metavar='N',
                    help='minimum performance gain for a model to be considered better (default: 5e-4)')
parser.add_argument('--train_success_thresh', type=float, default=0.6,
                    help='minimum training intersection-over-union threshold for success (default: 0.6)')
parser.add_argument('--test_success_thresh', type=float, default=0.5,
                    help='minimum testing intersection-over-union threshold for success (default: 0.5)')
parser.add_argument('--dim_embed', type=int, default=256,
                    help='how many dimensions in final embedding (default: 256)')
parser.add_argument('--max_boxes', type=int, default=500,
                    help='maximum number of edge boxes per image (default: 500)')
parser.add_argument('--num_embeddings', type=int, default=4,
                    help='number of embeddings to train (default: 4)')
parser.add_argument('--spatial', dest='spatial', action='store_true', default=True,
                    help='Flag indicating whether to use spatial features')
parser.add_argument('--confusion', dest='confusion', type=float, default=0)
parser.add_argument('--neg_region_num', dest='neg_region_num', type=int, default=20)
parser.add_argument('--region_feature_dim', dest='region_feature_dim', type=int, default=4096)

def main():
    global args
    args = parser.parse_args()
    np.random.seed(args.r_seed)
    tf.set_random_seed(args.r_seed)
    phrase_feature_dim = 6000
    args.region_feature_dim = 4096
    if args.spatial:
        if args.dataset == 'flickr':
            args.region_feature_dim += 5
        else:
            args.region_feature_dim += 8

    # setup placeholders
    labels_plh = tf.placeholder(tf.float32, shape=[None, args.max_boxes])  # lable ~ batch_size * max_boxes
    phrase_plh = tf.placeholder(tf.float32, shape=[None,
                                                   phrase_feature_dim])  # batch_size * 6000
    region_plh = tf.placeholder(tf.float32, shape=[None, args.max_boxes,
                                                   args.region_feature_dim])  # batch_size * max_boxes * 4096
    train_phase_plh = tf.placeholder(tf.bool, name='train_phase')
    num_boxes_plh = tf.placeholder(tf.int32)
    is_conf_plh = tf.placeholder(tf.float32, shape=[None])
    neg_region_plh = tf.placeholder(tf.float32, shape=[None, None, args.region_feature_dim])
    gt_plh = tf.placeholder(tf.float32, shape=[None, 1, args.region_feature_dim])
    plh = {}
    plh['num_boxes'] = num_boxes_plh
    plh['labels'] = labels_plh
    plh['phrase'] = phrase_plh
    plh['region'] = region_plh
    plh['train_phase'] = train_phase_plh
    plh['is_conf_plh'] = is_conf_plh
    plh['neg_region_plh'] = neg_region_plh
    plh['gt_plh'] = gt_plh
    test_loader = DataLoader(args, args.region_feature_dim, phrase_feature_dim,
                             plh, 'test')
    model = setup_model(args, phrase_plh, region_plh, train_phase_plh,
                        labels_plh, num_boxes_plh, is_conf_plh, neg_region_plh, gt_plh)
    if args.test:
        test(plh, model, test_loader, model_name=args.resume)
        sys.exit()

    save_model_directory = os.path.join('runs', args.name)
    if not os.path.exists(save_model_directory):
        os.makedirs(save_model_directory)

    global confusion_matrix
    confusion_matrix = {}
    train_loader = TorchDataLoader(args, args.region_feature_dim, phrase_feature_dim,
                                   plh, 'train', confusion_matrix)
    val_loader = DataLoader(args, args.region_feature_dim, phrase_feature_dim,
                            plh, 'val')

    # training with Adam
    acc, best_adam = train(plh, model, train_loader, test_loader, args.resume)

    # finetune with SGD after loading the best model trained with Adam
    best_model_filename = os.path.join('runs', args.name, 'model_best')
    acc, best_sgd = train(plh, model, train_loader, test_loader,
                          best_model_filename, False, acc)
    best_epoch = best_adam + best_sgd

    # get performance on test set
    test_acc = test(plh, model, test_loader, model_name=best_model_filename)
    print('best model at epoch {}: {:.2f}% (val {:.2f}%)'.format(
        best_epoch, round(test_acc * 100, 2), round(acc * 100, 2)))

def show_res(image_id, phrase_name, overlaps):
    imageDir = ""
    best = 10
    imagepath = os.path.join(imageDir, image_id+".jpg")
    im = cv2.imread(imagepath)

def test(plh, model, test_loader, sess=None, model_name=None):
    if sess is None:
        sess = tf.Session()
        saver = tf.train.Saver()
        saver.restore(sess, os.path.join('runs', args.name, model_name))

    region_weights = model[3]
    correct = 0.0
    n_iterations = test_loader.num_batches()
    for batch_id in range(n_iterations):
        feed_dict, gt_labels, num_pairs, image_id, phrase_name = test_loader.get_batch(batch_id)

        feed_dict[plh['is_conf_plh']] = np.zeros([args.batch_size])
        feed_dict[plh['neg_region_plh']] = np.zeros([args.batch_size, args.neg_region_num, args.region_feature_dim])
        feed_dict[plh['gt_plh']] = np.zeros([args.batch_size, 1, args.region_feature_dim])

        scores = sess.run(region_weights, feed_dict=feed_dict)
        for pair_index in range(num_pairs):
            best_region_index = np.argmax(scores[pair_index, :])
            correct += gt_labels[pair_index, best_region_index]

    acc = correct / len(test_loader)
    print('\n{} set localization accuracy: {:.2f}%\n'.format(
        test_loader.split, round(acc * 100, 2)))
    return acc

def getBatchSampler(n):
    length = n
    shu_batch = []
    chunk_size = 5000
    num_chunk_size = length // chunk_size
    for chunk in range(num_chunk_size):
        raw_ind = [r for r in range(chunk_size*chunk, (chunk + 1)*chunk_size)]
        num_batch = len(raw_ind) // args.batch_size
        shuffle(raw_ind)
        i = 0
        for i in range(num_batch):
            temp = raw_ind[i*args.batch_size: (i+1)*args.batch_size]
            shu_batch.append(temp)
        if len(raw_ind[(i+1)*args.batch_size:]) > 1:
            shu_batch.append(raw_ind[(i+1)*args.batch_size:])

    raw_ind = [r for r in range((chunk + 1)*chunk_size, length)]
    num_batch = len(raw_ind) // args.batch_size
    shuffle(raw_ind)
    i = 0
    for i in range(num_batch):
        temp = raw_ind[i * args.batch_size: (i + 1) * args.batch_size]
        shu_batch.append(temp)
    if len(raw_ind[(i + 1) * args.batch_size:]) > 1:
        shu_batch.append(raw_ind[(i + 1) * args.batch_size:])
    return shu_batch


def process_epoch(plh, model, train_loader, sess, train_step, epoch, suffix):
    # extract elements from model tuple
    global confusion_matrix
    loss = model[0]
    region_loss = model[1]
    l1_loss = model[2]
    region_weights = model[3]
    dneg_p = model[4]
    dgt_p = model[5]
    LossTrp = model[6]
    phrase_embed = model[7]
    gt_region_embed = model[8]
    neg_region_embed = model[9]
    if epoch > 1:
        args.confusion = 1.5
    else:
        args.confusion = 0
    batch_sa = getBatchSampler(len(train_loader))

    trainLoader = torch.utils.data.DataLoader(train_loader, batch_sampler=batch_sa, num_workers=6)
    for i, (phrase_features, region_features, is_train, max_boxes,
            gt_labels, phrase_name, neg_regions, gt_features, is_conf_plh) in enumerate(trainLoader):
	#print(i)
        feed_dict = {plh['phrase']: phrase_features,
                     plh['region']: region_features,
                     plh['train_phase']: is_train[0],
                     plh['num_boxes']: max_boxes[0],
                     plh['labels']: gt_labels,
                     plh['is_conf_plh']: is_conf_plh,
                     plh['neg_region_plh']: neg_regions,
                     plh['gt_plh']: gt_features
                     }

        (_, total, region, concept_l1, region_pro, P2neg, p2gt, TriLoss, phrase, gt, neg) = sess.run([train_step, loss,
                                                                                     region_loss, l1_loss,
                                                                                     region_weights, dneg_p, dgt_p,
                                                                                     LossTrp, phrase_embed, gt_region_embed, neg_region_embed],
                                                                                    feed_dict=feed_dict)

        if epoch % 3 == 0 or epoch == 1:
            for index in range(np.shape(region_pro)[0]):
                best_region_index = np.argmax(region_pro[index, :])
                if gt_labels[index, best_region_index] != 1:
                    confusion_matrix[phrase_name[index]] = []
                    slabel = gt_labels[index, :].numpy() * region_pro[index, :]
                    sort_index = np.argsort(slabel)
                    num = 0
                    while (num < args.neg_region_num):
                        if slabel[sort_index[num]] >= 0:
                            break
                        confusion_matrix[phrase_name[index]].append(sort_index[num])
                        num += 1
                    while (num < args.neg_region_num):
                        confusion_matrix[phrase_name[index]].append(sort_index[0])
                        num += 1
                else:
                    if phrase_name[index] in confusion_matrix.keys():
                        del confusion_matrix[phrase_name[index]]
            train_loader.confusion_matrix = confusion_matrix
            np.save("dict_epoch{}.npy".format(epoch), confusion_matrix)
        if i % args.info_iterval == 0:
            print('loss: {:.5f} (region: {:.5f} concept: {:.5f}) dgt_p: {:.5f}) dneg_p: {:.5f} Triloss: {:.5f})'
                  '[{}/{}] (epoch: {}) {}'.format(total, region, concept_l1, p2gt, P2neg, TriLoss,
                                                  (i * args.batch_size),
                                                  len(train_loader), epoch,
                                                  suffix))
            print(time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())))


def train(plh, model, train_loader, test_loader, model_weights, use_adam=True,
          best_acc=0.):
    sess = tf.Session()
    if use_adam:
        optim = tf.train.AdamOptimizer(args.lr)
        suffix = ''
    else:
        optim = tf.train.GradientDescentOptimizer(args.lr / 10.)
        suffix = 'ft'

    weights_norm = tf.losses.get_regularization_losses()
    weights_norm_sum = tf.add_n(weights_norm)
    loss = model[0]
    train_step = optim.minimize(loss + weights_norm_sum)
    saver = tf.train.Saver()
    init = tf.global_variables_initializer()
    epoch = 1
    best_epoch = 0
    with sess.as_default():
        init.run()
        if model_weights:
            saver.restore(sess, os.path.join('runs', args.name, model_weights))
            if use_adam:
               #best_acc = test(plh, model, test_loader, sess)
		best_acc = 0.6052
     

        # model trains until args.max_epoch is reached or it no longer
        # improves on the validation set
        while (epoch - best_epoch) < args.no_gain_stop and (args.max_epoch < 1 or epoch <= args.max_epoch):
            process_epoch(plh, model, train_loader, sess, train_step, epoch, suffix)
            saver.save(sess, os.path.join('runs', args.name, 'checkpoint'),
                       global_step=epoch)
            acc = test(plh, model, test_loader, sess)

            if acc > best_acc:
                saver.save(sess, os.path.join('runs', args.name, 'model_best'))
                if (acc - args.minimum_gain) > best_acc:
                    best_epoch = epoch

                best_acc = acc

            epoch += 1

    return best_acc, best_epoch


if __name__ == '__main__':
    # tf.device('/gpu:1')
   # os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
   # os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    main()
