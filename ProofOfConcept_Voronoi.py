from Utilities import *
from Models import *
from Losses import *
###############################################################################
def time_seed ():
    seed = None
    while seed == None:
        cur_time = time.time ()
        seed = int ((cur_time - int (cur_time)) * 1000000)
    return seed

class ImageDataFlow(RNGDataFlow):
    def __init__(self, 
        imageDir, 
        labelDir, 
        size, 
        dtype='float32', 
        isTrain=False, 
        isValid=False, 
        isTest=False, 
        pruneLabel=False, 
        skipped = 3, 
        shape=[3, 320, 320]):

        self.dtype      = dtype
        self.imageDir   = imageDir
        self.labelDir   = labelDir
        self._size      = size
        self.isTrain    = isTrain
        self.isValid    = isValid

        imageFiles = natsorted (glob.glob(self.imageDir + '/*.*'))
        labelFiles = natsorted (glob.glob(self.labelDir + '/*.*'))
        print(imageFiles)
        print(labelFiles)
        self.images = []
        self.labels = []
        self.data_seed = time_seed ()
        self.data_rand = np.random.RandomState(self.data_seed)
        self.rng = np.random.RandomState(999)
        for imageFile in imageFiles:
            image = skimage.io.imread (imageFile)
            if image.ndim==2:
                image = np.expand_dims(image, axis=0)
            self.images.append(image)
        for labelFile in labelFiles:
            label = skimage.io.imread (labelFile)
            if label.ndim==2:
                label = np.expand_dims(label, axis=0)
            self.labels.append(label)
            
        self.DIMZ = shape[0]
        self.DIMY = shape[1]
        self.DIMX = shape[2]
        self.pruneLabel = pruneLabel
        self.skipped = skipped
    def size(self):
        return self._size

    def AugmentPair(self, src_image, src_label, pipeline, seed=None, verbose=False):
        np.random.seed(seed) if seed else np.random.seed(2015)
        # print(src_image.shape, src_label.shape, aug_image.shape, aug_label.shape) if verbose else ''
        if src_image.ndim==2:
            src_image = np.expand_dims(src_image, 0)
            src_label = np.expand_dims(src_label, 0)
        
        # Create the result
        aug_images = [] #np.zeros_like(src_image)
        aug_labels = [] #np.zeros_like(src_label)
        
        # print(src_image.shape, src_label.shape)
        for z in range(src_image.shape[0]):
            #Image and numpy has different matrix order
            pipeline.set_seed(seed)
            aug_image = pipeline._execute_with_array(src_image[z,...]) 
            pipeline.set_seed(seed)
            aug_label = pipeline._execute_with_array(src_label[z,...])        
            aug_images.append(aug_image)
            aug_labels.append(aug_label)
        aug_images = np.array(aug_images).astype(np.float32)
        aug_labels = np.array(aug_labels).astype(np.float32)
        # print(aug_images.shape, aug_labels.shape)
        return aug_images, aug_labels
    ###############################################################################
    def random_reverse(self, image, seed=None):
        assert ((image.ndim == 2) | (image.ndim == 3))
        if seed:
            self.rng.seed(seed)
        random_reverse = self.rng.randint(1,3)
        if random_reverse==1:
            reverse = image[::1,...]
        elif random_reverse==2:
            reverse = image[::-1,...]
        image = reverse
        return image
    ###############################################################################
    def grow_boundaries(self, gt, steps=1, background=0):
        from scipy import ndimage
        foreground = np.zeros(shape=gt.shape, dtype=np.bool)
        masked = None
        
        for label in np.unique(gt):
            if label == background:
                continue
            label_mask = gt==label
            # Assume that masked out values are the same as the label we are
            # eroding in this iteration. This ensures that at the boundary to
            # a masked region the value blob is not shrinking.
            if masked is not None:
                label_mask = np.logical_or(label_mask, masked)
            eroded_label_mask = ndimage.binary_erosion(label_mask, iterations=steps, 
                                                       border_value=1)
            foreground = np.logical_or(eroded_label_mask, foreground)

        # label new background
        background = np.logical_not(foreground)
        gt[background] = 0
        
        return gt
    ###############################################################################
    def get_data(self):
        for k in range(self._size):
            #
            # Pick randomly a tuple of training instance
            #
            rand_index = self.data_rand.randint(0, len(self.images))
            image_p = self.images[rand_index]
            label_p = self.labels[rand_index]

            seed = time_seed () #self.rng.randint(0, 20152015)
            
                        # Downsample here
            #pz = self.data_rand.randint(0, 2)
            rs = 1 #self.data_rand.randint(1, self.skipped) # Random skip from 1, 2, 3

            py = self.data_rand.randint(0, rs)
            px = self.data_rand.randint(0, rs)
            image_p = image_p[::1, py::rs, px::rs].copy ()
            label_p = label_p[::1, py::rs, px::rs].copy ()
            # Cut 1 or 3 slices along z, by define DIMZ, the same for paired, randomly for unpaired

            dimz, dimy, dimx = image_p.shape
            # The same for pair
            randz = self.data_rand.randint(0, dimz-self.DIMZ+1)
            randy = self.data_rand.randint(0, dimy-self.DIMY+1)
            randx = self.data_rand.randint(0, dimx-self.DIMX+1)

            image_p = image_p[randz:randz+self.DIMZ,randy:randy+self.DIMY,randx:randx+self.DIMX]
            label_p = label_p[randz:randz+self.DIMZ,randy:randy+self.DIMY,randx:randx+self.DIMX]
            
            if self.isTrain:
                # Augment the pair image for same seed
                p_train = Augmentor.Pipeline()
                p_train.rotate_random_90(probability=0.75, resample_filter=Image.NEAREST)
                p_train.rotate(probability=1, max_left_rotation=20, max_right_rotation=20, resample_filter=Image.NEAREST)
                #p_train.random_distortion(probability=1, grid_width=4, grid_height=4, magnitude=5)
                p_train.flip_random(probability=0.75)

                image_p, label_p = self.AugmentPair(image_p.copy(), label_p.copy(), p_train, seed=seed)
                
                #image_p = self.random_reverse(image_p, seed=seed)
                #label_p = self.random_reverse(label_p, seed=seed)
                


            # # Calculate linear label
            if self.pruneLabel:
                label_p, nb_labels_p = skimage.measure.label(label_p.copy(), return_num=True)        

            # if self.grow_boundaries
            label_p[0,0,0] = 0 # hack for entire label is 1 due to CB3
            label_p = self.grow_boundaries(label_p)
            
            # Expand dim to make single channel
            image_p = np.expand_dims(image_p, axis=-1)
            label_p = np.expand_dims(label_p, axis=-1)

            # Calculate current mask and next mask
            list_values = np.unique(label_p)
            curr_values = list_values[:-1]
            next_values = list_values[1:]
            

            import random
            #curr_val = random.choice(curr_values)
            curr_idx = random.randrange(len(curr_values))
            curr_val = list_values[curr_idx]
            next_val = list_values[curr_idx+1]
            curr_mask = np.zeros_like(image_p)
            curr_mask[label_p<=curr_val] = 1.0
            curr_mask[label_p==0.0] = 0.0 # boundaries
            

            next_mask = np.zeros_like(image_p)
            next_mask[label_p==next_val] = 1.0
            yield [image_p.astype(np.float32), 
                   label_p.astype(np.float32), 
                   curr_mask.astype(np.float32), 
                   next_mask.astype(np.float32), 
                   ] 

###############################################################################
def get_data(dataDir, isTrain=False, isValid=False, isTest=False, shape=[16, 320, 320], skipped=3):
    # Process the directories 
    if isTrain:
        num=500
        names = ['trainA', 'trainB']
    if isValid:
        num=100
        names = ['trainA', 'trainB']
    if isTest:
        num=10
        names = ['validA', 'validB']

    
    dset  = ImageDataFlow(os.path.join(dataDir, names[0]),
                               os.path.join(dataDir, names[1]),
                               num, 
                               isTrain=isTrain, 
                               isValid=isValid, 
                               isTest =isTest, 
                               shape=shape, 
                               pruneLabel=True, 
                               skipped=skipped)
    dset.reset_state()
    return dset
###############################################################################
class Model(ModelDesc):
    @auto_reuse_variable_scope
    def generator(self, img, last_dim=1, nl=INLReLU3D, nb_filters=32):
        assert img is not None
        #img = tf.expand_dims(img, axis=0)
        ret = arch_fusionnet_translator_2d(img, last_dim=last_dim, nl=nl, nb_filters=nb_filters)
        #ret = tf.squeeze(ret, axis=0)
        return ret 

    def inputs(self):
        return [
            tf.placeholder(tf.float32, (args.DIMZ, args.DIMY, args.DIMX, 1), 'image'),
            tf.placeholder(tf.float32, (args.DIMZ, args.DIMY, args.DIMX, 1), 'label'),
            tf.placeholder(tf.float32, (args.DIMZ, args.DIMY, args.DIMX, 1), 'curr_'),
            tf.placeholder(tf.float32, (args.DIMZ, args.DIMY, args.DIMX, 1), 'next_'),
            ]

    def build_graph(self, image, label, curr_, next_):
        G = tf.get_default_graph()
        pi, pl, pc, pn = image, label, curr_, next_

        # Construct the graph
        with tf.variable_scope('gen'):
            with tf.device('/device:GPU:0'):
                with tf.variable_scope('image2first'):
                    pin = self.generator(tf.concat([tf_2tanh(pi, maxVal=255.0), tf_2tanh(pc, maxVal=1.0)], axis=-1), 
                                            last_dim=1, nl=tf.nn.tanh, nb_filters=32)
                    pin = tf_2imag(pin, maxVal=1.0)
        pin = tf.identity(pin, 'pin')
       
        losses = []            
        with tf.name_scope('loss_mae'):
            mae_in = tf.reduce_mean(tf.abs(pin - pn), name='mae_in')
            losses.append(1e0*mae_in)
            add_moving_summary(mae_in)

        with tf.name_scope('loss_dice'):
            dice_in = tf.identity(1.0 - dice_coe(pin, pn, axis=[0,1,2,3], loss_type='jaccard'), 
                                 name='dice_in')  
            #dice_in = tf.where(tf.is_nan(dice_in), tf.ones_like(dice_in) * 1e+2, dice_in);
            losses.append(1e2*dice_in)
            add_moving_summary(dice_in)


        # Aggregate final loss
        self.cost = tf.reduce_sum(losses, name='self.cost')
        add_moving_summary(self.cost)

        # Segmentation
        pz = tf.zeros_like(pi)
        viz = tf.concat([tf.concat([pi, 10*pl, 255*pc, 255*pn, 255*pin], axis=2),
                         #tf.concat([pl, 255*pif[...,0:1], 255*pif[...,1:2], 255*pif[...,2:3]], axis=2),
                         #tf.concat([pz, 255*pa [...,0:1], 255*pa [...,1:2], 255*pa [...,2:3]], axis=2),
                         #tf.concat([pil*255, 255*pia[...,0:1], 255*pia[...,1:2], 255*pia[...,2:3]], axis=2),
                         ], axis=1)
        viz = tf.cast(tf.clip_by_value(viz, 0, 255), tf.uint8, name='viz')
        tf.summary.image('labelized', viz, max_outputs=50)


    def optimizer(self):
        lr = symbolic_functions.get_scalar_var('learning_rate', 2e-4, summary=True)
        return tf.train.AdamOptimizer(lr, beta1=0.5, epsilon=1e-3)

###############################################################################
class VisualizeRunner(Callback):
    def __init__(self, input, tower_name='InferenceTower', device=0):
        self.dset = input 
        self._tower_name = tower_name
        self._device = device

    def _setup_graph(self):
        self.pred = self.trainer.get_predictor(
            ['image', 'label', 'curr_', 'next_'], ['viz'])

    def _before_train(self):
        pass

    def _trigger(self):
        for lst in self.dset.get_data():
            viz_test = self.pred(lst)
            viz_test = np.squeeze(np.array(viz_test))
            self.trainer.monitors.put_image('viz_test', viz_test)

###############################################################################


def sample(dataDir, model_path, prefix='.'):    
    pass
###############################################################################
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu',        default='0', help='comma seperated list of GPU(s) to use.')
    parser.add_argument('--data',  default='data/Kasthuri15/3D/', required=True, 
                                    help='Data directory, contain trainA/trainB/validA/validB')
    parser.add_argument('--load',   help='Load the model path')
    parser.add_argument('--DIMX',  type=int, default=256)
    parser.add_argument('--DIMY',  type=int, default=256)
    parser.add_argument('--DIMZ',  type=int, default=20)
    parser.add_argument('--SKIP',  type=int, default=4)
    parser.add_argument('--sample', help='Run the deployment on an instance',
                                    action='store_true')
    global args
    args = parser.parse_args()
    
    # python Exp_FusionNet2D_-VectorField.py --gpu='0' --data='arranged/'

    
    train_ds = get_data(args.data, isTrain=True, isValid=False, isTest=False, shape=[args.DIMZ, args.DIMY, args.DIMX], skipped=args.SKIP)
    valid_ds = get_data(args.data, isTrain=False, isValid=True, isTest=False, shape=[args.DIMZ, args.DIMY, args.DIMX], skipped=args.SKIP)
    # test_ds  = get_data(args.data, isTrain=False, isValid=False, isTest=True)


    train_ds  = PrefetchDataZMQ(train_ds, 4)
    train_ds  = PrintData(train_ds)
    # train_ds  = QueueInput(train_ds)
    model     = Model()

    os.environ['PYTHONWARNINGS'] = 'ignore'

    # Set the GPU
    if args.gpu:
        os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

    # Running train or deploy
    if args.sample:
        # TODO
        print("Deploy the data")
        sample(args.data, args.load, prefix='hdims_')
        # pass
    else:
        # Set up configuration
        # Set the logger directory
        logger.auto_set_dir()

        # Set up configuration
        config = TrainConfig(
            model           =   model, 
            dataflow        =   train_ds,
            callbacks       =   [
                PeriodicTrigger(ModelSaver(), every_k_epochs=200),
                PeriodicTrigger(VisualizeRunner(valid_ds), every_k_epochs=5),
                ScheduledHyperParamSetter('learning_rate', [(0, 2e-4), (100, 1e-4), (200, 2e-5), (300, 1e-5), (400, 2e-6), (500, 1e-6)], interp='linear'),
                ],
            max_epoch       =   1000, 
            session_init    =   SaverRestore(args.load) if args.load else None,
            )
    
        # Train the model
        launch_train_with_config(config, QueueInputTrainer())
