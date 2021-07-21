## Classify Fake or Real Images

##Approach A
In file *"classifier_conv32_drop0.5.png"* is shown the model attached at the end of the Encoder model and then inside *"train_classifier_log_classifier1.csv"* there are the metrics/statistics of the learning process. The metrics that are used are AUC, Precision, Recall, accuracy, loss for validation and train sets.

##Approach B
Check file *"classifier_conv64_drop.2.png"* where you can see the architecture of the model stacked with the Encoder and inside *"train_classifier_log_classifier2.csv"* you can observe the whole learning/training process for 70 epochs and batch size =128. Still the same metrics are been used.

##Results:
From the two approaches the B (second) one performs better, lookin at the last line inside *"train_classifier_log_classifier2.csv"* -> trainAUC =0.99,trainAcc=0.99,trainPrecision=0.99,trainRecall=0.99.



## Notice:
both approaches using the same batchSize and number of Epochs.
