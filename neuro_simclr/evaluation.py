import sklearn 

def evaluate_linear_readout(backbone, train_dataset, test_dataset, args):
    """
        Evaluate the linear readout
    """
    # Embed each of the neuroid data using the backbone
    train_embeddings = []
    train_labels = []
    for data in train_dataset:
        neuroid_data, labels = data
        neuroid_data = neuroid_data.to(args.device)
        train_embeddings.append(
            backbone(neuroid_data).detach().cpu().numpy()
        )
        train_labels.append(labels)
    # Embed the test data
    test_embeddings = []
    test_labels = []
    for data in test_dataset:
        neuroid_data, labels = data
        neuroid_data = neuroid_data.to(args.device)
        test_embeddings.append(
            backbone(neuroid_data).detach().cpu().numpy()
        )
        test_labels.append(labels)
    # Train a logistic classifier on those data pairs 
    # similar to what is done in `linear_readout_baseline`
    clf = sklearn.linear_model.LogisticRegressionCV().fit(
        train_embeddings,
        train_labels
    )
    # Evaluate the model on the test set
    accuracy = clf.score(
        test_embeddings,
        test_labels
    )
    # Predict f1 score
    predicted_labels = clf.predict(test_embeddings)
    fscore = sklearn.metrics.f1_score(test_labels, predicted_labels)

    return accuracy, fscore
