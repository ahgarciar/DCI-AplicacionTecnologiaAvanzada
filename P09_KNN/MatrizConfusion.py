
def confusionMatrix(y_true, y_pred, distinctClass):
    clasesName = ["Class_" + str(i) for i in range(distinctClass)]
    from sklearn.metrics import classification_report
    result = classification_report(y_true, y_pred, target_names=clasesName, output_dict=True)
    #print(result)
    return result["accuracy"]
