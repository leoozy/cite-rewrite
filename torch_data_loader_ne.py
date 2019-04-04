import numpy as np
import h5py
import os
import pdb
import time
from torch.utils.data import Dataset


class TorchDataLoader(Dataset):
    """Class minibatches from data on disk in HDF5 format"""

    def __init__(self, args, region_dim, phrase_dim, plh, split, confusion_matrix = None):
        """Constructor

        Arguments:
        args -- command line arguments passed into the main function
        region_dim -- dimensions of the region features
        phrase_dim -- dimensions of the phrase features
        plh -- placeholder dictory containing the tensor inputs
        split -- the data split (i.e. 'train', 'test', 'val')
        """

        self.dataset = None
        self.datafn = os.path.join('/temphome/zhangjl', '%s_imfeats.h5' % split)
        with h5py.File(self.datafn, 'r', swmr=True) as dataset:

            vecs = np.array(dataset['phrase_features'], np.float32)
            phrases = list(dataset['phrases'])
            assert (vecs.shape[0] == len(phrases))

            w2v_dict = {}
            for index, phrase in enumerate(phrases):
                w2v_dict[phrase] = vecs[index, :]

            # mapping from uniquePhrase to w2v
            self.w2v_dict = w2v_dict
            self.pairs = list(dataset['pairs'])
        self.n_pairs = len(self.pairs[0])
        self.pair_index = range(self.n_pairs)
        self.split = split
        self.plh = plh
        self.is_train = split == 'train'
        self.neg_to_pos_ratio = args.neg_to_pos_ratio
        self.batch_size = args.batch_size
        self.max_boxes = args.max_boxes
        self.gtFeaturePath = os.path.join('/media/zhangjl/ZJLSSD/', 'gt_feature')
        self.args = args
        if self.is_train:
            self.success_thresh = args.train_success_thresh
        else:
            self.success_thresh = args.test_success_thresh

        self.region_feature_dim = region_dim
        self.phrase_feature_dim = phrase_dim

        if not confusion_matrix is None:
            self.confusion_matrix = confusion_matrix

    def __len__(self):
        return self.n_pairs

    def __getitem__(self, index):
        """Returns a minibatch given a valid id for it

        Arguments:
        batch_id -- number between 0 and self.num_batches()

        Returns:
        feed_dict -- dictionary containing minibatch data
        gt_labels -- indicates positive/negative regions
        num_pairs -- number of pairs without padding
        """

        with h5py.File(self.datafn, 'r', swmr=True) as dataset:

            region_features = np.zeros((self.max_boxes,
                                        self.region_feature_dim), dtype=np.float32)

            gt_labels = np.zeros((self.max_boxes),
                                 dtype=np.float32)
            phrase_features = np.zeros((self.phrase_feature_dim),
                                       dtype=np.float32)

            # print("index", index)
            # paired image
            im_id = self.pairs[0][index]

            # paired phrase
            phrase = self.pairs[1][index]

            # phrase instance identifier
            p_id = self.pairs[2][index]
           # start1 = time.time()
            # gets region features
            features = np.array(dataset[im_id], np.float32)
            #start2 = time.time()
            num_boxes = min(len(features), self.max_boxes)
            #start3 = time.time()
            spatial = features[0, -9:]
            xmin = spatial[-4]
            ymin = spatial[-3]
            W = xmin / spatial[-9]
            H = ymin / spatial[-8]

            features = features[:num_boxes, :self.region_feature_dim]
           # start4 = time.time()
            overlaps = np.array(dataset['%s_%s_%s' % (im_id, phrase, p_id)])
            gt_box = overlaps[-4:]
            spatial_feature = [gt_box[0]/W, gt_box[1]/H, gt_box[2]/W, gt_box[3]/H, (gt_box[3]-gt_box[1])*(gt_box[2] - gt_box[0])/(W*H)]
            #start5 = time.time()
            # last 4 dimensions of overlaps are ground truth box coordinates
            assert (num_boxes <= len(overlaps) - 4)
            overlaps = overlaps[:num_boxes]
            region_features[:num_boxes, :] = features
            phrase_features[:] = self.w2v_dict[phrase]
            gt_labels[:num_boxes] = overlaps >= self.success_thresh
            #start6 = time.time()
            neg_regions = np.zeros([self.args.neg_region_num, self.region_feature_dim])
            #gt_features = np.zeros([1,self.region_feature_dim])
            is_conf_plh = self.args.confusion

         #   print(time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())))
            if self.is_train:
                num_pos = int(np.sum(gt_labels[:]))
                num_neg = num_pos * self.neg_to_pos_ratio
                negs = np.random.permutation(np.where(overlaps < 0.3)[0])
                phrase_name = '%s_%s_%s' % (im_id, phrase, p_id)
                if len(negs) < num_neg:  # if not enough negatives
                    negs = np.random.permutation(np.where(overlaps < 0.4)[0])

                # logistic loss only counts a region labeled as -1 negative
                gt_labels[negs[:num_neg]] = -1


        if self.args.confusion != 0.:
            if phrase_name in self.confusion_matrix.keys():

                for id, neg_region_id in enumerate(self.confusion_matrix[phrase_name]):
                            neg_regions[id] = region_features[neg_region_id, :]
                try:
                    gt_features = np.load(os.path.join(self.gtFeaturePath, phrase_name+'.npy'))
                    gt_features = np.append(gt_features, spatial_feature)
                    gt_features = np.expand_dims(gt_features, axis= 0)

                except Exception, e:
                    print(e)

            else:
                neg_regions = np.zeros([self.args.neg_region_num, self.region_feature_dim])
                gt_features = np.zeros([1, self.region_feature_dim])
        else:
            neg_regions = np.zeros([self.args.neg_region_num, self.region_feature_dim])
            gt_features = np.zeros([1, self.region_feature_dim])
            is_conf_plh = 0.


        return phrase_features, region_features, self.is_train, self.max_boxes, gt_labels, phrase_name, neg_regions, gt_features, is_conf_plh
        #return phrase_features, region_features, self.is_train, self.max_boxes, gt_labels


