## Sagemaker Fraud Detection Workshop

### Lab description

This lab demonstrates three different ML algorithms used for identifying fraudelent transactions on the same dataset:
- SageMaker XGBoost
- AutoEncoders
- Neural Networks

### Steps for launching the workshop environment using EVENT ENGINE 
Note: these steps were tested on Chrome browser using Mac OS
#### open a browser and navigate to https://dashboard.eventengine.run/login
#### Enter a 12-character "hash" provided to you by workshop organizer.
#### Click on "Accpet Terms & Login"
![Navigate to Sagemaker Service](/images/image-01.png)

#### Click on "AWS Console"
![Navigate to Sagemaker Service](/images/image-02.png)

#### Please, log off from any other AWS accounts you are currently logged into

#### Click on "Open AWS Console"
![Navigate to Sagemaker Service](images/image-03.png)

#### You should see a screen like this. 
#### We now need select the correct Identity Role for the workshop
#### Type "IAM" into the search bar and click on IAM 
(Identity and Access Management).
![Navigate to Sagemaker Service](/images/image-04.png)

#### Click on "Roles"
![Navigate to Sagemaker Service](/images/image-05.png)

#### Scroll down past "Create Role" and Click on "TeamRole"
![Navigate to Sagemaker Service](/images/image-06.png)

#### Copy "Role ARN" by selecting the copy icon on the right
#### You may want to temporariliy paste this role ARN into a notepad 
#### Once you copied TeamRole ARN, click on "Services" in the upper left corner
![Navigate to Sagemaker Service](/images/image-07.png)

#### Enter "SageMaker" in the search bar and click on it
![Navigate to Sagemaker Service](/images/image-08.png)

#### You should see a screen like this. 
#### Click on the orange button "Create Notebook Instance"
![Navigate to Sagemaker Service](/images/image-09.png)

#### On the next webpage, 
#### - Give your notebook a name (no underscores, please)
#### - Under Notebook instance type, select "ml.c5.2xlarge"
#### - Under "Permission and encryption" select "Enter a custom IAM role ARN";
#### - Paste your TeamRole ARN in the cell below labled "Custom IAM role ARN"
####      Note: your TeamRole ARN will have different AWS account number than what you see here
#### - Scroll down to the bottom of the page and click on "Create Notebook instance"
![Navigate to Sagemaker Service](/images/image-10.png)

#### You should see your notebook being created. In a couple of minutes, its status will change
#### from "Pending" to "In Service", at which point, please click on "Open Jupyter"
![Navigate to Sagemaker Service](/images/image-11.png)

#### In Jupyter Notebook console, please, click on 'New' -> 'Terminal' on the right-hand side
![Navigate to Sagemaker Service](/images/image-12.png)

#### A new Chrome browser tab will open displaying a command prompt terminal
#### In the terminal tap, please, issue these two commands:
####    $ cd SageMaker 
####    $ git clone https://github.com/aws-samples/amazon-sagemaker-fraud-detection
#### You should see output similar to this:
![Navigate to Sagemaker Service](/images/image-13.png)

#### You may now close the browser tab with command prompt terminal,
#### return to Jupyter console and navigate the created folder structure to
#### amazon-sagemaker-fraud-detection -> notebooks
#### launch and run each one of the three Jupyter notebooks
![Navigate to Sagemaker Service](/images/image-14.png)











#### Open SageMaker Console by clicking on "Services" and searching for Sagemaker
![Navigate to Sagemaker Service](/images/image-08.png)

## License

This library is licensed under the MIT-0 License. See the LICENSE file.

