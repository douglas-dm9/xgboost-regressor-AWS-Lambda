### Project Brief

You have been hired as a data scientist at Discount Motors, a used car dealership in the UK. The dealership is expanding and has hired a large number of junior salespeople. Although promising, these junior employees have difficulties pricing used cars that arrive at the dealership. Sales have declined 18% in recent months, and management would like your help designing a tool to assist these junior employees.

To start with, they would like you to work with the Toyota specialist to test your idea(s). They have collected some data from other retailers on the price that a range of Toyota cars were listed at. It is known that cars that are more than £1500 above the estimated price will not sell. The sales team wants to know whether you can make predictions within this range.

Main goals in this project:

- Build a regression model to estimated toyota car's price with a RMSE less than £1500.
- Deploy the final model to the cloud

#### Instructions to deploy

- Install all the dependent librarys in the file requirements.txt
- Run the script train.py to train the final model and saving it to bentoml 
- Run the script test.py to serve the model locally with bentoml
- Create the bentofile.yaml and build your bento
- Follow the inscrutions to deploy using bentoctl in this link  [https://github.com/bentoml/bentoctl](link)

#### [Final model deployed on AWS Lambda](https://6dipvpzn4j.execute-api.sa-east-1.amazonaws.com/#/Service%20APIs/xgboost_regressor__xgboost_regressor)

Example of one single prediction: 

Data: {"model":"GT86","year":2016,"transmission":"Manual","mileage":24089,"fuelType":"Petrol","tax":265,"mpg":36.2,"engineSize":2.0}
![image](https://user-images.githubusercontent.com/58889801/208963180-5190f0b1-18c0-4d0c-bbe4-07280fddeb20.png)
