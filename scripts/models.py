from xgboost import XGBClassifier
from sklearn.ensemble import GradientBoostingClassifier

class SelectionClassifier(GradientBoostingClassifier):
    def __init__(self, query=[], **kwargs):
        print('Initialized! Biatches')
        self.query = ' '.join(query)
        super(GradientBoostingClassifier, self).__init__(**kwargs)

    def get_params(self):
        print(super(GradientBoostingClassifier, self).get_params(deep=True))
        return super(GradientBoostingClassifier, self).get_params(deep=True)
