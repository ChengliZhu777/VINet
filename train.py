from utils.general import set_logging, get_base_options

if __name__ == '__main__':
    set_logging(-1)

    train_options = get_base_options(device_indices='0',
                                     workers=4,
                                     resume=False)
    
