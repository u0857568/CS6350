python id3.py -train car/train.csv -test car/test.csv -car -depth 1
python id3.py -train car/train.csv -test car/test.csv -car -depth 2
python id3.py -train car/train.csv -test car/test.csv -car -depth 3
python id3.py -train car/train.csv -test car/test.csv -car -depth 4
python id3.py -train car/train.csv -test car/test.csv -car -depth 5
python id3.py -train car/train.csv -test car/test.csv -car -depth 6

python id3.py -train car/train.csv -test car/test.csv -car -depth 1 -ME
python id3.py -train car/train.csv -test car/test.csv -car -depth 2 -ME
python id3.py -train car/train.csv -test car/test.csv -car -depth 3 -ME
python id3.py -train car/train.csv -test car/test.csv -car -depth 4 -ME
python id3.py -train car/train.csv -test car/test.csv -car -depth 5 -ME
python id3.py -train car/train.csv -test car/test.csv -car -depth 6 -ME

python id3.py -train car/train.csv -test car/test.csv -car -depth 1 -GI
python id3.py -train car/train.csv -test car/test.csv -car -depth 2 -GI
python id3.py -train car/train.csv -test car/test.csv -car -depth 3 -GI
python id3.py -train car/train.csv -test car/test.csv -car -depth 4 -GI
python id3.py -train car/train.csv -test car/test.csv -car -depth 5 -GI
python id3.py -train car/train.csv -test car/test.csv -car -depth 6 -GI

python id3.py -train bank/train.csv -test bank/test.csv -bank -depth 1
python id3.py -train bank/train.csv -test bank/test.csv -bank -depth 4
python id3.py -train bank/train.csv -test bank/test.csv -bank -depth 8
python id3.py -train bank/train.csv -test bank/test.csv -bank -depth 12
python id3.py -train bank/train.csv -test bank/test.csv -bank -depth 16

python id3.py -train bank/train.csv -test bank/test.csv -bank -depth 1 -ME
python id3.py -train bank/train.csv -test bank/test.csv -bank -depth 4 -ME
python id3.py -train bank/train.csv -test bank/test.csv -bank -depth 8 -ME
python id3.py -train bank/train.csv -test bank/test.csv -bank -depth 12 -ME
python id3.py -train bank/train.csv -test bank/test.csv -bank -depth 16 -ME

python id3.py -train bank/train.csv -test bank/test.csv -bank -depth 1 -GI
python id3.py -train bank/train.csv -test bank/test.csv -bank -depth 4 -GI
python id3.py -train bank/train.csv -test bank/test.csv -bank -depth 8 -GI
python id3.py -train bank/train.csv -test bank/test.csv -bank -depth 12 -GI
python id3.py -train bank/train.csv -test bank/test.csv -bank -depth 16 -GI

python id3.py -train bank/train.csv -test bank/test.csv -bank -depth 1 -missing
python id3.py -train bank/train.csv -test bank/test.csv -bank -depth 4 -missing
python id3.py -train bank/train.csv -test bank/test.csv -bank -depth 8 -missing
python id3.py -train bank/train.csv -test bank/test.csv -bank -depth 12 -missing
python id3.py -train bank/train.csv -test bank/test.csv -bank -depth 16 -missing

python id3.py -train bank/train.csv -test bank/test.csv -bank -depth 1 -ME -missing
python id3.py -train bank/train.csv -test bank/test.csv -bank -depth 4 -ME -missing
python id3.py -train bank/train.csv -test bank/test.csv -bank -depth 8 -ME -missing
python id3.py -train bank/train.csv -test bank/test.csv -bank -depth 12 -ME -missing
python id3.py -train bank/train.csv -test bank/test.csv -bank -depth 16 -ME -missing

python id3.py -train bank/train.csv -test bank/test.csv -bank -depth 1 -GI -missing
python id3.py -train bank/train.csv -test bank/test.csv -bank -depth 4 -GI -missing
python id3.py -train bank/train.csv -test bank/test.csv -bank -depth 8 -GI -missing
python id3.py -train bank/train.csv -test bank/test.csv -bank -depth 12 -GI -missing
python id3.py -train bank/train.csv -test bank/test.csv -bank -depth 16 -GI -missing


python adaboost.py -train bank/train.csv -test bank/test.csv -bank -depth 1 -T 5