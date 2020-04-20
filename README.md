# Generating Dogs using the DCGAN architecture

## Data

- The data is available at http://vision.stanford.edu/aditya86/ImageNetDogs/
- The model graph is available at http://download.tensorflow.org/models/image/imagenet/inception-2015-12-05.tgz

The data expects to be put in an `input` directory at the root and organized the following way :

```bash
input
├── all-dogs
│   ├── dog_image_1.jpg
│   ├── dog_image_2.jpg
│   └── ...
├── classify_image_graph_def.pb
├── Annotation
    ├── dog_breed_1
    ├── dog_breed_2
    └── ...
```
