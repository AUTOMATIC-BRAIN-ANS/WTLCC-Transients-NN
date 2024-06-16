import torchvision.transforms as tvtransforms

AVAILABLE_TRANSFORMS = {
    "ToTensor": tvtransforms.ToTensor
}


def get_transforms(config):
    list_of_transforms = []
    if config["transforms"] is not None:
        for i in range(len(config["transforms"])):
            transform_name = config["transforms"][i]
            transform_params = config["transform_params"][transform_name]
            if transform_params is not None:
                list_of_transforms.append(AVAILABLE_TRANSFORMS[transform_name](**transform_params))
            else:
                list_of_transforms.append(AVAILABLE_TRANSFORMS[transform_name]())
        
        return tvtransforms.Compose(list_of_transforms)
