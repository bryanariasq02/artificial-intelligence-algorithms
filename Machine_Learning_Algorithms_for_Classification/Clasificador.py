def KNN(X_train, Y_train, X_test, Y_test, k=3):
    print("\n\tKNN")
    from sklearn.neighbors import KNeighborsClassifier
    Modelo_0 = KNeighborsClassifier(k)
    Modelo_0.fit(X_train, Y_train)
    Y_pred_0 =Modelo_0.predict (X_test)
    # Graficando resultados
    # print('Mostrando el Expected y predicted')
    # for j in range(10): 
    #     print (Y_test[j],' ',Y_pred_0[j])
    #     print('')

    from sklearn.metrics import confusion_matrix
    matrix = confusion_matrix(Y_test, Y_pred_0)
    print(matrix)
    print('')

    from sklearn.metrics import accuracy_score,precision_score
    print("Accuracy ",accuracy_score(Y_test, Y_pred_0))