import bentoml 
from bentoml.io import JSON
import numpy as np

bento_model = bentoml.xgboost.get('xgboost_regressor:latest')
pre_processor = bento_model.custom_objects["pre_processor"]
scaler = bento_model.custom_objects["scaler"]

model_runner = bento_model .to_runner()


svc = bentoml.Service("xgboost_regressor",runners=[model_runner])

@svc.api(input=JSON(), output=JSON())
def xgboost_regressor(application_data):
    application_data=np.array([list((pre_processor(application_data)).values())])
    prediction=float(model_runner.predict.run(scaler.transform(application_data))[0])
    return {"Veicule estimated value":round(prediction,2)}