import os
import sys
import pandas as pd
from src.exception import CustomException
from src.utils import load_object



class PredictPipiline:
    def __init__(self):
        pass


    def predict(self,features):
        try:
            model_path=os.path.join("artifacts","model.pkl")
            preprocessor_path=os.path.join('artifacts','preprocessor.pkl')
            print("Before Loading")
            model=load_object(file_path=model_path)
            preprocessor=load_object(file_path=preprocessor_path)
            print("After Loading")
            data_scaled=preprocessor.transform(features)
            preds=model.predict(data_scaled)
            return preds
        
        except Exception as e:
            raise CustomException(e,sys)



class CustomData:
    def __init__(
            self, 
            batting_team: str,
            bowling_team: str,
            city: str,
            current_score,
            overs,
            wickets,
            last_five: int,
    ):
        
        self.batting_team = batting_team

        self.bowling_team = bowling_team
        
        self.city = city
        
        self.current_score = int(current_score)

        self.overs = int(overs)

        self.wickets = int(wickets)

        self.last_five = last_five

        self.balls_left = 120 - (self.overs * 6)

        self.wickets_left = 10 - self.wickets
        
        self.crr =  self.current_score / self.overs
    
    def get_data_as_data_frame(self):
        try:
            custom_data_input_dict = {
                'batting_team': [self.batting_team],
                'bowling_team': [self.bowling_team],
                'city': [self.city],
                'current_score': [self.current_score],
                'balls_left': [self.balls_left],
                'wickets_left': [self.wickets_left],
                'crr' : [self.crr],
                'last_five' : [self.last_five]
            }

            return pd.DataFrame(custom_data_input_dict)
        
        except Exception as e:
            raise CustomException(e, sys)
    