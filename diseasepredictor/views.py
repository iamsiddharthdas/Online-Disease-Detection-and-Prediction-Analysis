from django.shortcuts import render
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier


def bmi(request):
    value=''
    if request.method == 'POST':
        gender=float(request.POST['gender'])
        height=float(request.POST['height'])
        weight=float(request.POST['weight'])

        bmiv=(weight * 10000.0) / ( height * height)
        #print(bmi)
        if( bmiv < 18.5):
            value= "UnderWeight:("
        elif(bmiv >= 18.5) and (bmiv < 24.9) :
            value= "Normal:)"
        elif(bmiv >= 25) and (bmiv < 29.9):
            value= "Overweight:("
        elif(bmiv >= 30 ):
            value= "Obese:("
        else:
            value="Please enter a Valid Input.."

    return render(request,
                  'bmi.html',
                  {
                      'context': value,
                      'title': 'Body Mass Index(Health Status)',
                      'active': 'btn btn-success peach-gradient text-violet',
                      'bmi': True,
                      'background': 'bg-secondary text-white '
                  })


def heart(request):
    """
    Reading the training data set.
    """
    df = pd.read_csv('static/Heart_train.csv')
    data = df.values
    X = data[:, :-1]
    Y = data[:, -1:]

    """
    Reading data from the user.
    """

    value = ''

    if request.method == 'POST':

        age = float(request.POST['age'])
        sex = float(request.POST['sex'])
        cp = float(request.POST['cp'])
        trestbps = float(request.POST['trestbps'])
        chol = float(request.POST['chol'])
        fbs = float(request.POST['fbs'])
        restecg = float(request.POST['restecg'])
        thalach = float(request.POST['thalach'])
        exang = float(request.POST['exang'])
        oldpeak = float(request.POST['oldpeak'])
        slope = float(request.POST['slope'])
        ca = float(request.POST['ca'])
        thal = float(request.POST['thal'])

        user_data = np.array(
            (age,
             sex,
             cp,
             trestbps,
             chol,
             fbs,
             restecg,
             thalach,
             exang,
             oldpeak,
             slope,
             ca,
             thal)
        ).reshape(1, 13)

        rf = RandomForestClassifier(
            n_estimators=16,
            criterion='entropy',
            max_depth=9
        )

        rf.fit(np.nan_to_num(X), Y)
        rf.score(np.nan_to_num(X), Y)
        predictions = rf.predict(user_data)

        if int(predictions[0]) == 1:
            value = 'You have:('
        elif int(predictions[0]) == 0:
            value = "You don\'t have:)"

    return render(request,
                  'heart.html',
                  {
                      'context': value,
                      'title': 'Heart Disease Prediction',
                      'active': 'btn btn-success peach-gradient text-white',
                      'heart': True,
                      'background': 'bg-danger text-white'
                  })


def diabetes(request):
    """
    Reading the training data set.
    """
    dfx = pd.read_csv('static/Diabetes_XTrain.csv')
    dfy = pd.read_csv('static/Diabetes_YTrain.csv')
    X = dfx.values
    Y = dfy.values
    Y = Y.reshape((-1,))

    """
    Reading data from user.
    """
    value = ''
    if request.method == 'POST':

        pregnancies = float(request.POST['pregnancies'])
        glucose = float(request.POST['glucose'])
        bloodpressure = float(request.POST['bloodpressure'])
        skinthickness = float(request.POST['skinthickness'])
        bmi = float(request.POST['bmi'])
        insulin = float(request.POST['insulin'])
        pedigree = float(request.POST['pedigree'])
        age = float(request.POST['age'])

        user_data = np.array(
            (pregnancies,
             glucose,
             bloodpressure,
             skinthickness,
             bmi,
             insulin,
             pedigree,
             age)
        ).reshape(1, 8)

        knn = KNeighborsClassifier(n_neighbors=3)
        knn.fit(X, Y)

        predictions = knn.predict(user_data)

        if int(predictions[0]) == 1:
            value = 'You have:('
        elif int(predictions[0]) == 0:
            value = "You don\'t have:)"

    return render(request,
                  'diabetes.html',
                  {
                      'context': value,
                      'title': 'Diabetes Disease Prediction',
                      'active': 'btn btn-success peach-gradient text-white',
                      'diabetes': True,
                      'background': 'bg-brown text-white'
                  }
                  )


def breast(request):

    df = pd.read_csv('static/Breast_train.csv')
    data = df.values
    X = data[:, :-1]
    Y = data[:, -1]
    print(X.shape, Y.shape)


    value = ''
    if request.method == 'POST':

        radius = float(request.POST['radius'])
        texture = float(request.POST['texture'])
        perimeter = float(request.POST['perimeter'])
        area = float(request.POST['area'])
        smoothness = float(request.POST['smoothness'])

        rf = RandomForestClassifier(
            n_estimators=16, criterion='entropy', max_depth=5)
        rf.fit(np.nan_to_num(X), Y)

        user_data = np.array(
            (radius,
             texture,
             perimeter,
             area,
             smoothness)
        ).reshape(1, 5)

        predictions = rf.predict(user_data)
        print(predictions)

        if int(predictions[0]) == 1:
            value = 'You have:('
        elif int(predictions[0]) == 0:
            value = "You don\'t have:)"

    return render(request,
                  'breast.html',
                  {
                      'context': value,
                      'title': 'Breast Cancer Prediction',
                      'active': 'btn btn-success peach-gradient text-white',
                      'breast': True,
                      'background': 'bg-pink text-white'
                  })


def home(request):

    return render(request,
                  'home.html')


"""
Handling 404 error pages.
"""


def handler404(request):
    return render(request, '404.html', status=404)
