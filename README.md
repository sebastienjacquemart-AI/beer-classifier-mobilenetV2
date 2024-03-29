# Beer Classifier

If you are reading this, you are probably applying for a Machine Learning engineering job at X. This test will evaluate if you have the right skills for the job. The time needed to complete this challenge depends on your experience and can range from half a day to multiple tries over a week.

In this test, you will try to classify Belgium draft beers. If you are able to complete this test in a decent way, you might soon be drinking one of these together with us.

You can start by cloning this repository to a local and/or private location via:

```bash
git clone https://bitbucket.org/ml6team/challenge-classify-draft-beer.git
```

## The Data

Before you begin to implement your classification model you need to download the data used for training and local evaluation from Google Cloud Storage and place the `data` folder in the base folder. To download the data execute the following command (you will need to [install the `gsutil` command](https://cloud.google.com/storage/docs/gsutil_install#sdk-install) beforehand which is part of the *Google Cloud SDK*):

```bash
gsutil -m cp -R gs://ml6_junior_ml_engineer_challenge_cv_beer_data/data .
```

For your purposes, the data has already been split into training data and evaluation data. They are respectively in the `train` folder and `eval` folder. In each you will find four folders which represent the beers you'll need to classify. There are five kinds of draft beer: Chimay Blue, Orval, Rochefort 10, Westmalle Tripel and Westvleteren 12. The classes are numbered 0 to 4 respectively. These class numbers are necessary to create a correct classifier. If you want, you can inspect the data, however, the code to load the images into NumPy arrays is already written for you.


## The Models

In the `trainer` folder, you will be able to see several Python files. The `data.py`, `task.py` and `final_task.py` files are already coded for you. The only file that needs additional code is the `model.py` file. The comments in this file will indicate which code has to be written.

To test how your model is doing you can execute the following command (you will need to [install](https://cloud.google.com/sdk/docs/#install_the_latest_cloud_sdk_version) the `gcloud` command, which is also part of the *Google Cloud SDK*):

```bash
python -m trainer.task
```

If you run this command before you wrote any code in the `model.py` file, you will notice that it returns errors. Your goal is to write code that does not return errors and achieves an accuracy that is as high as possible.

The command above will train and evaluate your classifier. The batch size and the number of epochs need to be defined in the `model.py` file.

Make sure you'll think about the solution you will submit for this coding test. If you want you can change the code written by us according to your needs. It is however important that we can still perform our automated evaluation when you submit your solution so make sure you test your solution thoroughly before you submit it. How you can test your solution is explained below.

![Data overview](data.png)

The command above uses the `task.py` file. As you can see in the figure above, this file only uses the beer images in the `train` folder and uses the images in the `eval` folder to evaluate the model. This is excellent to test how the model performs but to obtain a better evaluation one can also train upon all available data which should increase the performance on the dataset you will be evaluated on. After you finished your solution in `model.py`, you can continue reading to learn how to train your model on the full dataset.


## Deploying the Model

Once you've got the code working you will need to deploy the model to Google Cloud to turn it into an API that can receive new images of draft beers and returns its prediction them. Don't worry, the code for this is already written in the `final_task.py` file. To deploy your model, you only have to run a few commands in your command line.

To export your trained model and to train your model on the images in the `train` and `eval` folder you have to execute the following command (only do this once you've completed coding the `model.py` file):

```bash
python -m trainer.final_task
```

Once you've executed this command, you will notice that the `output` folder was created in the root directory of this repository. This folder contains your saved model that you'll need to deploy to Google Cloud Vertex AI.

In order to do so you will need to create a [Google Cloud account](https://cloud.google.com/). You will need a credit card for this, but you'll get [free credit from Google](https://cloud.google.com/free/docs/gcp-free-tier/#free-trial) to run your Vertex AI instance. **Note:** if you are not eligible for the free trial please reach out to us as you are responsible for the costs associated with the project.

Once you've created your Google Cloud account, you'll need to import your model into Vertex AI using this [guide](https://cloud.google.com/vertex-ai/docs/model-registry/import-model). Next, you'll have to deploy your model to an endpoint. For this, you can follow this [guide](https://cloud.google.com/vertex-ai/docs/text-data/classification/get-predictions#deploy_a_model_to_an_endpoint). Make sure to deploy the model using Tensorflow 2.9 and we advice to use **`europe-west1` as the regional endpoint** if available. Note that the region of the bucket should correspond to the region of the model.


## Checking your Deployed Model

Before you submit your solution, you can check if your deployed model works correctly by executing the following commands:

```bash
ENDPOINT_ID=<your_endpoint_id>
PROJECT_ID=<your_project_id>

gcloud ai endpoints predict $ENDPOINT_ID \
    --project=$PROJECT_ID  \
    --region=europe-west1 \
    --json-request=check_deployed_model/test.json
```

Check if you are able to get a prediction out of the `gcloud` command. If you get errors, you should try to resolve them before submitting the solution. The output of the command should look something like this (the numbers will probably be different):

```
CLASSES  PROBABILITIES
1        [0.004145291168242693, 0.9800060987472534, 0.004468264523893595, 0.007732450030744076, 0.0036478929687291384]
```

The value for the `$ENDPOINT_ID` variable can be found in your project on the Google Cloud web interface. You will need this value and your Google Cloud *Project ID* to submit your coding test.

To be able to pass the coding test. You should be able to get an accuracy of 80% on our secret dataset of beers (which you don't have access to). If your accuracy however seems to be less than 80% after we evaluated it, you can just keep submitting solutions until you are able to get an accuracy of 80%.


## Submitting your Coding Test

Once you are able to execute the command above without errors, you can add us to your project:

* Go to the menu of your project
* Click *IAM & admin*
* Click *Add*
* Add `ml6-coding-challenge-evaluator@recruiting-220608.iam.gserviceaccount.com` as a member with the role *Project Owner*

After you added us to your project you should fill in [this form](https://docs.google.com/forms/d/e/1FAIpQLScW_j-QmmqUo6W_yyUH2xovSQ5kcAF3HapNep5pk5nr-nXKwQ/viewform) so we are able to automatically evaluate your solution to the coding test. Once you've filled in the form you should receive an email with the results within 2 hours. We'll hope with you that your results are good enough to land an interview at X. If however you don't you can resubmit a new solution as many times as you want, so don't give up!

If you are invited for an interview at X afterwards, make sure to bring your laptop with a copy of the code you wrote, so you can explain your `model.py` file to us.

### Taking down the deployed model

After you have received the evaluation email, we no longer require access to the model. Please check the corresponding documentation page for removing Vertex AI models. This can be done via the UI or the command line.

# Image classification

source 1: https://www.ml6.eu/blogpost/effective-image-labeling-for-computer-vision-projects


In our experience with projects involving labeled data there is one constant truth: having a well-annotated training set is crucial to success. As the saying goes: garbage in, garbage out.

To get a good grip on what your training set needs to look like, you have to consider what problem you’re actually trying to solve. How different are the various things you want to be able to detect? Is there a lot of variation in how they look, how they are lit, the angle the image is taken from? Is one camera type used, or multiple different models?

While various techniques exist to make the most of annotating on a limited time budget, it is still good practice to assemble a training set that covers images covering all of the above variety that applies to your problem. A machine learning model will have an easier time generalizing what it knows to a new example closely resembling to what it was trained on, rather than something entirely novel.

Before labeling, do a pass over the dataset to spot patterns that could make for some difficult decisions in your labeling approach. Did you think of all relevant categories? Is there a certain type of object that is in between two categories? Do you have some partially visible objects in some images? Is it actually impossible to recognise some objects in certain conditions (e.g. dark objects in dark images) Think about these questions and decide on an appropriate plan beforehand. If your labeling is inconsistent, the quality of the model trained on your data will be negatively affected. You might need to leave out images that are not clear enough. A good rule of thumb is using your own eye: if you can make a split-second decision on a visual task, a machine learning model can usually learn to replicate it. However, if enough visual information is just not present in the image, the model will not be able to pull it from thin air.

source 1: https://www.ml6.eu/blogpost/how-to-discover-good-data-augmentation-policies-for-ml

Andrew Ng, the co-founder and former head of Google Brain, is spearheading a shift in AI research and usage towards what he coined “Data-Centric AI” and away from the practice of “Model-Centric AI” that has dominated research over the years.

“machine learning has matured to the point that high-performance model architectures are widely available, while approaches to engineering datasets have lagged.”

Machine learning is an iterative process: make changes, train, evaluate and repeat. Consequently it is important to have a good starting point, so that subsequent iterations can be compared to that baseline to quickly figure out what works and what does not. This approach also applies to data augmentation.

One particularly interesting development in that regard is Google’s AutoAugment. They formulate the problem of finding the best augmentation policy for a dataset as a discrete search problem.

Finding such an optimal augmentation policy for a sufficiently large search space requires a lot of compute power. Because of this, running AutoAugment on your own dataset is not a viable option. But, there’s good news. The authors of AutoAugment argue that learnt policies are transferable between datasets, e.g. an optimal policy for ImageNet performs well on other datasets similar to ImageNet.

e can see an obvious parallel to transfer learning where pretrained weights from one dataset often produce good results on other datasets as well.

If you thought waving the term Data-centric AI around would excuse you completely from having to understand your model, you are mistaken. Some models are developed with a specific data preparation in mind. Disregarding this, might negatively impact your model performance.

Other (coding) sources: 

https://www.tensorflow.org/tutorials/images/transfer_learning

Binary Image Classification with transfer learning

https://www.tensorflow.org/tutorials/keras/classification

Binary Image Classification with model from scratch

https://huggingface.co/docs/transformers/tasks/image_classification

Image Classification using transfer learning (transformers)

https://huggingface.co/models?pipeline_tag=image-classification&sort=trending

Trending Image Classification models on Hugging Face

https://bitbucket.org/ml6team/challenge-classify-draft-beer/src/master/

# Theory (TBC)

In machine learning, the loss function quantifies the model's performance, gradient descent optimizes this function by adjusting the weights to minimize the loss, which are parameters that determine the behavior of the model.

Training metrics are used to optimize model parameters during the training phase, while evaluation metrics assess the model's performance on unseen data; overfitting occurs when a model learns to memorize training data but performs poorly on new data, whereas generalization refers to a model's ability to perform well on unseen data after training.

A pre-trained model is a saved network that was previously trained on a large dataset, typically on a large-scale image-classification task. You either use the pretrained model as is or use transfer learning to customize this model to a given task. You can then take advantage of these learned feature maps without having to start from scratch by training a large model on a large dataset.

To generate predictions from the block of features, average over the spatial 5x5 spatial locations, using a tf.keras.layers.GlobalAveragePooling2D layer to convert the features to a single 1280-element vector per image. Apply a tf.keras.layers.Dense layer to convert these features into a single prediction per image.

Fine-Tuning: Unfreeze a few of the top layers of a frozen model base and jointly train both the newly-added classifier layers and the last layers of the base model.

When you don't have a large image dataset, it's a good practice to artificially introduce sample diversity by applying random, yet realistic, transformations to the training images, such as rotation and horizontal flipping. This helps expose the model to different aspects of the training data and reduce overfitting.

You will create the base model from the MobileNet V2 model developed at Google. This is pre-trained on the ImageNet dataset, a large dataset consisting of 1.4M images and 1000 classes. ImageNet is a research training dataset with a wide variety of categories like jackfruit and syringe.

Accuracy is a commonly used metric in image classification tasks because it provides a straightforward measure of how well the model is performing. However, accuracy may not always be the most suitable metric, especially in scenarios where the classes are imbalanced.

Epochs refer to the number of times the entire dataset is passed forward and backward through a neural network during training, while batch size determines the number of samples processed in each iteration. Increasing the number of epochs without regularization techniques may lead to overfitting by allowing the model to memorize the training data excessively. Regularization techniques, such as L1 and L2 regularization, prevent the weights in a neural network from becoming too large by adding a penalty term to the loss function. This penalty discourages the model from learning overly complex patterns that might result in memorizing the training data rather than learning generalizable patterns. Dropout is a regularization technique commonly used in neural networks
