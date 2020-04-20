# Generating Dogs Using the DCGAN Architecture

For further explaination about the project consider checking the report, `report.pdf`.

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
└── Annotation
    ├── dog_breed_1
    ├── dog_breed_2
    └── ...
```

## Code

The code is in the `src` directory. To run experiments, use the notebook in the 'notebook' directory

## Experiments

Runs were saved as a `.html` files to share the results of different set-ups. Check `DCGAN Experiments.xlsx` for an overview of the experiments.
