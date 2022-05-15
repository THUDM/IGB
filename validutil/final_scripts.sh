echo "GCN on Cora"
python gcn.py --dataset=Cora --lr 0.005 --hidden 32 --dropout 0.1 --weight_decay 0.0001 --runs 10
echo "GCN on CiteSeer"
python gcn.py --dataset=CiteSeer --lr 0.01 --hidden 64 --dropout 0.3 --weight_decay 0.0001 --runs 10
echo "GCN on PubMed"
python gcn.py --dataset=PubMed --lr 0.01 --hidden 64 --dropout 0.3 --weight_decay 0.0001 --runs 10

echo "APPNP on Cora"
python appnp.py --dataset=Cora --lr 0.01 --hidden 64 --dropout 0.6 --weight_decay 0.0001 --K 8 --alpha 0.3 --runs 10
echo "APPNP on CiteSeer"
python appnp.py --dataset=CiteSeer --lr 0.01 --hidden 64 --dropout 0.3 --weight_decay 0.0001 --K 6 --alpha 0.3 --runs 10
echo "APPNP on PubMed"
python appnp.py --dataset=PubMed --lr 0.01 --hidden 256 --dropout 0.3 --weight_decay 0.0001 --K 8 --alpha 0.3 --runs 10

echo "MixHop on Cora"
python mixhop.py --dataset=Cora --lr 0.001 --dropout 0.2 --weight_decay 5e-5 --layer1_pows 200 200 200 --layer2_pows 100 100 100 --runs 10
echo "MixHop on CiteSeer"
python mixhop.py --dataset=CiteSeer --lr 0.001 --dropout 0.8 --weight_decay 0.003 --layer1_pows 120 120 120 --layer2_pows 60 60 60 --runs 10
echo "MixHop on PubMed"
python mixhop.py --dataset=PubMed --lr 0.001 --dropout 0.3 --weight_decay 3e-4 --layer1_pows 80 80 80 --layer2_pows 40 40 40 --runs 10

