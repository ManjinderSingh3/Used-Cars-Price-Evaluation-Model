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
There are several Machine learning algorithms to perform regression tasks such as linear regression, SVM Regressor, Random Forest regressor, etc. After analysing features present in the dataset and comparing results of various regression models, I have found that Random Forest Regressor model has provided best results with an **R2_Score of 92.4**

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
  - Click on Inbound Rules -> Edit Inbound Rules.
  - For Port 22, allow it to everyone's IP address as the source. As this application is not having such security concerns so I have allowed it for everyone. If you are dealing with some secured information than select My IP as the source. This ensures that only your IP can remotely connect to the EC2 instance.
- Click Review and Launch and Launch the instance.
- For connecting the instance securely, create a new key pair and download the private key file. Keep this file safe as this will enable anyone to connect the EC2 instance. For Mac uses `.pem` file will directly work, however, for Windows users `.pem` file needs to be converted to `.ppk` file 

## Connect to the EC2 Instance
- Right-click on the running EC2 instance on the AWS management console and click connect.
- Follow the instructions to connect to the EC2 instance remotely from your command line.

## Copy Code to EC2 Instance
In order to get the code on EC2 instance we have two basic ways:
a. Clone the code from Github repository
b. Copy the Code to AWS S3 bucket from your local system and than copy the project directory from AWS S3 to the EC2 instance.

### a. Clone the Project from Github repository
- Coneecting to EC2 instance from local system as per steps mentioned above.
- Clone the complete project with `git clone https://github.com/ManjinderSingh3/Used-Cars-Price-Evaluation-Model.git` on EC2 instance.
- `cd <project-directory>`
- Install pip3 and required dependencies using the below commands.
  ```
  sudo apt-get update

  sudo apt-get -y install python3-pip

  pip3 install -r requirements.txt 
  ```
- Run the `app.py` file using below mentioned command
  ```
  python3 app.py
  ```
- You will now be able to access the  web app with https:EC2-IP:PORT. (Port number which I have used is 8080)

### b. Copy Project to S3 Bucket
- Install AWS CLI. It is used to interact with the AWS console from command line. Follow the instructions [here](https://docs.aws.amazon.com/cli/latest/userguide/getting-started-install.html)
- Follow instructions [here](https://aws.amazon.com/premiumsupport/knowledge-center/create-access-key/) to get an AWS access key.
- Configure AWS CLI by typing `aws configure` from your command line/ terminal.
- Provide the Access Key Id, Access Key, and Default Region Name.
- Now create an AWS S3 Bucket to store the project using below mentioned command.
  ```
  aws s3 mb s3://bucket-name
  ```
- Copy files from your local project directory to S3 bucket using below mentioned command.
  ```
  aws s3 cp <your directory path> s3://<your bucket name> --recursive
  ```
- Now the project is copied to S3 bucket. We have to now provide S3 access from EC2 instance. To enable access follow the instructions [here](https://aws.amazon.com/premiumsupport/knowledge-center/ec2-instance-access-s3-bucket/)
- Copy the project directory from AWS S3 to the EC2 instance using below mentioned command.
  ```
  aws s3 sync <local directory path> s3://source-bucket-name
  ```
- Switch the directory in which project is copied using `cd <project-directory>`.
- Install pip3 and required dependencies using the below commands.
  ```
  sudo apt-get update

  sudo apt-get -y install python3-pip

  pip3 install -r requirements.txt 
  ```
- Run the `app.py` file using below mentioned command
  ```
  python3 app.py
  ```
- You will now be able to access the  web app with https:EC2-IP:PORT. (Port number which I have used is 8080)

# Contact 📞

#### If you have any doubt or want to contribute to this project feel free to email me or drop your message on [LinkedIn](https://www.linkedin.com/in/manjinder-singh-a23aa3149/)


