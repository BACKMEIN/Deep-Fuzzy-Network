class hparams:

    train_or_test = 'train'
    output_dir = ''
    aug = True
    latest_checkpoint_file = ''
    total_epochs = 300
    epochs_per_checkpoint = 50
    batch_size = 8
    ckpt = None
    init_lr = 0.001
    scheduer_step_size = 20
    scheduer_gamma = 0.8
    debug = False
    mode = '2d' # '2d or '3d'
    in_class = 1
    out_class = 1
    crop_or_pad_size = 512,384,1 # if 3D: 256,256,256

    fold_arch = '*.'

    source_train_images_0_dir = ''
    source_train_images_1_dir = ''
    source_train_labels_0_dir = ''
    source_train_labels_1_dir = ''
    source_test_images_0_dir = ''
    source_test_images_1_dir = ''
    source_test_labels_0_dir = ''
    source_test_labels_1_dir = ''





