git clone --single-branch -b orig-code https://github.com/dev-chauhan/PQG-pytorch.git PQG
wget https://ndownloader.figshare.com/files/11931260?private_link=5463afb24cba05629cdf
mkdir PQG/data
mv ./11931260\?private_link\=5463afb24cba05629cdf ./PQG/data/quora_data_prepro.json
mkdir pretrained