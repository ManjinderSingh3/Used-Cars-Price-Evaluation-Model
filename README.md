# Used Cars Price Prediction Model 🚗

## This project lies under Regression Category of Supervised Machine Learning. 
Technologies used in building this project are:
- Programming Language : Python
- Frameworks/Libraries : Dash and Plotly

Note: This project is also deployed on AWS EC2 instance. Cloud Deployment steps are mentioned in subsequent section of this file.

The market for used cars is increasing tremendously these days. Hence, there is a need for a prediction system that tells the price of a used car using a variety of features. This model will help buyers as well as sellers in quoting honest price for their vehicle.

# Dataset 📊
### Source: Kaggle

The dataset used to implement this project can be found [here](https://www.kaggle.com/austinreese/craigslist-carstrucks-data).
This dataset includes the record of every used vehicle within the United States. It includes details like the price of the vehicle, manufacturing company details, current vehicle condition, fuel type and 22 other columns which capture every detail of the vehicle.

# Exploratory Data Analysis 
1. Visualizing Dataset
![image](https://github.com/ManjinderSingh3/Used-Cars-Price-Evaluation-Model/blob/master/results/1.png)
2. Average Price of Vehicles based on Fuel Type
![image](https://github.com/ManjinderSingh3/Used-Cars-Price-Evaluation-Model/blob/master/results/2.png)

3. Percentage of Vehicles per Manufacturer
![image](https://github.com/ManjinderSingh3/Used-Cars-Price-Evaluation-Model/blob/master/results/3.png)

4. Number of Cars available in market at different price ranges
![image](https://github.com/ManjinderSingh3/Used-Cars-Price-Evaluation-Model/blob/master/results/4.png)

5. Availablity of Used Cars over the years
![image](https://github.com/ManjinderSingh3/Used-Cars-Price-Evaluation-Model/blob/master/results/5.png)
6. Correlation among columns 
![image](https://github.com/ManjinderSingh3/Used-Cars-Price-Evaluation-Model/blob/master/results/6.png)
7. Number of Cars based on paint color
![image](https://github.com/ManjinderSingh3/Used-Cars-Price-Evaluation-Model/blob/master/results/7.png)
8. Number of Cars available in market based on Category of Vehicle
![image](https://github.com/ManjinderSingh3/Used-Cars-Price-Evaluation-Model/blob/master/results/8.png)

# Machine Learning Model 💻
There are several Machine learning algorithms to perform regression tasks such as linear regression, SVM Regressor, Random Forest regressor, etc. After analysing features present in the dataset and comparing results of various regression models, I have found that Random Forest Regressor model has provided best results with an R2_Score of 92.4

## Random Forest Regressor
Some important parameters while building the regressor are:
- Number of estimators: 100 (By default this algorithm use 100 trees and I got good results with this default value)
- After data pre-processing, there were 160,000 records. Among them I have used 80% of the data to train the model and  rest 20% for testing with a random state of 42.
- I used  R2_Score, Mean Square Error, and Mean Absolute Error to evaluate the performance of the model.

# Dashboard to Predict Price 
![image](https://github.com/ManjinderSingh3/Used-Cars-Price-Evaluation-Model/blob/master/results/9.png)

## How Price Prediction will take place 💡
- Users choose several parameters from the prediction dashboard as per their choice.
- Upon choosing all the parameters, model will convert the user inputs into the format (normalized data) in which model is trained. I have created a dictionary where the original value is mapped with normalized value.
- After pre-processing user inputs, these values are passed to the saved model to make price predictions.
- Model is saved with the help of pickle package. For every new input I am using saved model to perform prediction which prevent the training step at every iteration.

# Feature Importance 💥
After analysing the dataset, I have used 10 prominent features which will decide the price of used cars. Among these 10 features, every feature is not equally important. Some of the features have more weightage while others have less weightage in calculating the price. To identify the importance of each feature I have used the feature_importances_ parameter of the model. This parameter returns the weightage of each feature.
![image](https://github.com/ManjinderSingh3/Used-Cars-Price-Evaluation-Model/blob/master/results/10.png)

From the above figure, it is evident that Manufacturing Year has the highest impact in calculating the price followed by Odometer rating. Year column/feature alone has a weightage of 41% which is quite significant. Drive (4wd, fwd, rwd) has also a good impact in deciding price. Rest of the features have less than 10% weightage.

# Advance Price Suggestion 🔦
While testing the model, I got R2_Score of 92.4%. So, to fix this 10% error rate, model will provide an advance feature where user will get a broad price range which they can consider while buying the vehicle. Range of 10% on the upper side and 10% on the lower side of the Predicted price will be available to customer's to help them in making decision.
![image](https://github.com/ManjinderSingh3/Used-Cars-Price-Evaluation-Model/blob/master/results/11.png)

# How to run code locally 🛠️
- Before following below mentioned steps please make sure you have [git](https://git-scm.com/download), [Anaconda](https://www.anaconda.com/) or [pycharm](https://www.jetbrains.com/pycharm/download) and [git-lfs](https://git-lfs.github.com) installed on your system.
- Clone the complete project with `git clone https://github.com/ManjinderSingh3/Used-Cars-Price-Evaluation-Model.git` or you can just download the code and unzip it.
- Install all libraries mentioned in requirement.txt file
- As the pickle file is greater than 100 Mb so I have used git-lfs for this file. While clonning if you don't have git-lfs installed on your system than run `prediction_models.py` file to build the model and save it as .pkl file.
- Now, finally run the project with
  ```
  python app.py
  ```
- Open the localhost url provided after running `app.py` and now you can use the project locally in your web browser.

# Deployment on AWS ☁️

## Create an EC2 Instance
- Create an EC2 t2.micro instance as the server for this web app. From the AWS management console, under services, click on EC2.
- Select Ubuntu Server (latest version) as the Amazon Machine Image (AMI).
- Select t2.micro instance type, which is free tier eligible. 
- Configure Security Groups by adding inbound rules.
- - Click on Inbound Rules -> Edit Inbound Rules.
- - For Port 22, allow it to everyone's IP address as the source. As this application is not having such security concerns so I have allowed it for everyone. If you are dealing with some secured information than select My IP as the source. This ensures that only your IP can remotely connect to the EC2 instance.
- Click Review and Launch and Launch the instance.


# Contact 📞

#### If you have any doubt or want to contribute to this project feel free to email me or drop your message on [LinkedIn](https://www.linkedin.com/in/manjinder-singh-a23aa3149/)


