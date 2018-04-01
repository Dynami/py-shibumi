global_params = {
    'db_path':'/Users/Dacia/Documents/02.Ale/py-projects/py-shibumi/historical_data.db',
    'rnn_model_path':'./models/run/model_mix76.h5',
    'rnn_model_weights_path':'./models/run/model_mix76_weights.h5',
    'hmm_model_path':'./models/run/hmm_model.pkl',
    
    'save_models':True,
    'models_dir':'./models/'
}
symbols = [
    #{'symbol': '^GSPC'},
    {'symbol': 'GOOG'}, #0.7146950470732706
    {'symbol': 'AAPL'}, #0.7208350388866148
    {'symbol': 'AIG'},  #0.7191977077363897
    {'symbol': 'MS'}, #0.7322963569381907
    {'symbol': 'NKE'}, #0.7122390503479329
    {'symbol': 'C'}, #0.7265656979124028

    {'symbol': 'AMZN'}, #0.7339336880884159
    {'symbol': 'MSFT'}, #0.7146950470732706
    {'symbol': 'JPM'}, #0.7122390503479329
    {'symbol': 'CSCO'}, #0.7204257060990585
    {'symbol': 'KO'}, #0.703643061809251
    {'symbol': 'MCD'}, #0.711011051985264
    {'symbol': 'INTC'}, #0.7335243553008596
    {'symbol': 'TSLA'} #0.7331168831168832
]