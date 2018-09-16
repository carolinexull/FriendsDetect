Determine whether two people are friends based on their recent comments in [Yelp dataset](https://www.yelp.com/dataset/).<br>
training set: 9000 pairs<br>
test set: 1000 pairs<br>
we take the concatenated comments of two people as input.


* 200 words each person, length of inputs: 400

|    model    |       acc       |
|:------------:|:---------------:| 
|   BiLSTM  |     0.737    |
| BiLSTM+Att|     0.761    |
|   Att    |      0.750   |