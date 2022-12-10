#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Dec  4 12:18:39 2022

@author: sandeepvemulapalli
"""

import streamlit as st
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
import altair as alt
import numpy as np
import plotly.express as px
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score,confusion_matrix
from sklearn.metrics import classification_report

from xgboost import XGBClassifier
from PIL import Image

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score, GridSearchCV

from sklearn.cluster import KMeans

siteHeader = st.container()
dataExploration = st.container()
newFeatures = st.container()

with siteHeader:
    st.title('Welcome to the  ML Web App!')

with dataExploration:
    st.header('Dataset: Body performance')

data = pd.read_csv("bodyPerformance.csv")


# Data PreProcessing

data_copy = data.copy()



data['gender'].replace(['F','M'],[0,1],inplace=True)
data['class'].replace(['A','B','C','D'],[0,1,2,3],inplace=True)

data_cols = data.columns.to_list()


data_outlier = data.copy()




def class_cat(cls):
    if cls == 0:
        return 0
    if cls == 1:
        return 0
    if cls == 2:
        return 1
    if cls == 3:
        return 1


data_outlier["class"] = data_outlier["class"].astype(int)
data_outlier["cls_cat"] = data_outlier["class"].apply(class_cat)


x2 = data_outlier.copy()




indexsitbend = x2[(x2['sit and bend forward_cm'] <= 0) | (x2['sit and bend forward_cm'] >= 120) ].index
x2 = x2.drop(indexsitbend)

indexsystolic= x2[(x2['systolic'] <= 0) | (x2['systolic'] >= 180) ].index
x2 = x2.drop(indexsystolic)

y = x2["cls_cat"] #defined y

x2 = x2.drop(columns=["class","cls_cat"])


with st.sidebar:
    st.write("Which option do you want to select ?")
    
    
    d= st.radio(label='select one',options=['Dataset','EDA','PCA','Feature Engineering','HyperParameter Tuning','Models'])
    
    
if(d=="Dataset"):
    
    image = Image.open("fitness.png")
    st.image(image,caption="Image taken from Spanish Fan share- Google")
    
    st.header("Motivation")
    st.write("Maintaining a healthy lifestyle, which includes eating the right foods  \n", 
             " exercising, and avoiding junk food, will not only help us live longer and have younger-looking skin and hair,  " 
            " but it will also improve our overall wellbeing. As a result, we'll feel better both physically and mentally.")
    
    
    st.dataframe(data_copy)
    st.write("This data is originally from Korea sports  Promotion foundation and it  "
" contains the following attributes. \n")
    st.markdown("1.Age:21-64")
    st.markdown("2.gender: Female,Male")
    st.markdown("3.height in cms ")
    st.markdown(" 4.weight in kgs")
    st.markdown("5.body fat Percentage ")
    st.markdown(" 6.diastolic : diastolic blood pressure")
    st.markdown("7.systolic : systolic blood pressure ")
    st.markdown("8.gripForce ")
    st.markdown("9.sit and bend forward_cms " )
    st.markdown("10.sit-ups counts")
    st.markdown("11.broad jump_cm")
    st.markdown("12.class : A,B,C,D ")
    
    
elif (d =='EDA'):
    
    #st.dataframe(data_copy)
    
    st.text('Scatter plots')
        
    selected_option_1= st.selectbox("Select an attribute for x",data_cols )
    selected_option_2= st.selectbox("Select an attribute for y",data_cols )
        
        
    chart_1 = alt.Chart(data_copy).mark_circle().encode(
        x=selected_option_1,
        y=selected_option_2,
        color='gender').interactive()

    fig1=st.altair_chart((chart_1).interactive(),use_container_width=True)
    
    
######HISTOGRAM#########  
 
    st.subheader("Histogram")
    
    selected_option_3= st.selectbox("Select an option to plot",data_cols,  )
    
    age_s = st.number_input('Please select the age ', 21.0, 64.0,value=50.0)
    data_new = data[data_copy['age']<= age_s]
    
    bar = alt.Chart(data_new).mark_bar().encode(
    x='age',
    y=selected_option_3)
    fig2=st.altair_chart((bar).interactive(),use_container_width=True)
    
    
### Violin Plots####

    #sns.violinplot(data=data_copy, x="gender", y=selected_option_3)
    
    
    
    st.subheader("Violin plots")
    fig = px.violin(data_copy, x="class",y="height_cm",color="gender")
    st.plotly_chart(fig)
    #st.write("1. The average weight of males is higher than females")
    st.subheader("Insights")
    st.markdown("1. The average weight of males is higher than females.  \n"
" 2. The average fat percentage is also higher in males compared to females.  \n"
" 3. The average stretch length in males is higher than females.  \n"
"  4. However, body fat percentage is higher in females than males.")
    
    
### Other violin plots
    st.subheader("Class wise distribution")
    fig_sitbend_cm = px.violin(data_copy, x="class",y='sit and bend forward_cm',color="class")
    st.plotly_chart(fig_sitbend_cm)
    


    
    st.subheader("Count plot with class wise breakdown")
    fig= plt.figure(figsize=(4,6))
    sns.countplot(x=data_copy['class'],hue=data_copy['gender'])
    st.pyplot(fig)
    st.markdown("1. Consequently, males are more fit than females. But this is not true since"
                "the proportion of males and females in dataset is not same.")
    

    bp_chart=alt.Chart(data_copy).mark_point().encode(
    x='sit and bend forward_cm',
    y='broad jump_cm',
    color='gender',
    tooltip=['age','gripForce','sit and bend forward_cm']
    ).interactive()

    fig1=st.altair_chart((bp_chart).interactive(),use_container_width=True)   

    bp_chart_2=alt.Chart(data_copy).mark_point().encode(
     x='weight_kg',
     y='body fat_%',
     color='gender',
     tooltip=['age','gripForce','sit and bend forward_cm']
     ).interactive()

    fig1=st.altair_chart((bp_chart_2).interactive(),use_container_width=True)
    
    st.write("From the above figure, we can see that for a given weight, females have "
             " higher body fat percentage compated to males.")
   
    
elif (d=='Feature Engineering'):
    
    st.write("Initially, after doing the EDA, I observed there were lot of outliers "
             " in the data especially in systolic and sit and bend forward. I removed these "
             " outliers so that model can fit the data better." ) 
    
    fig_out_1 = px.violin(data_outlier, x="class",y='sit and bend forward_cm',color="class")
    st.plotly_chart(fig_out_1)
    st.write("lets remove some outliers in the data")
    

    
    indexsitbend = data_outlier[(data_outlier['sit and bend forward_cm'] <= 0) | (data_outlier['sit and bend forward_cm'] >= 120) ].index
    data_outlier = data_outlier.drop(indexsitbend)
    st.write(data_outlier)
    
    fig_out_2 = px.violin(data_outlier, x="class",y='sit and bend forward_cm',color="class")
    st.plotly_chart(fig_out_2)
    
    st.write("This plot is after removing the outliers in the data")
    
    indexsystolic= data_outlier[(data_outlier['systolic'] <= 0) | (data_outlier['systolic'] >= 180) ].index
    data_outlier = data_outlier.drop(indexsystolic)
    
    fig_out_3 = px.violin(data_outlier, x="class",y='sit and bend forward_cm',color="class")
    st.plotly_chart(fig_out_3)
  
    st.write("I realised that the model was not accurate enough if 4 classes are there. Hence "
             " I combined the class 0 and 1 to 0. Additionally, combined 2,3 to 1. Now, the accuracy "
             " of the model increased drastically.")
  
    st.subheader("Improving the accuracy")
    
    
    
    
    
    
    
    
    data_new = data_outlier.drop(columns=["class"])
    
    
    x_out = data_new.drop(columns=["cls_cat"])
    
    #st.write(x_out.shape)
    
    y_out = data_new["cls_cat"]
    
    # Scaling the data and applying XGboost
    
    scaler = StandardScaler()
        
    X_out=scaler.fit_transform(x_out)

    XgbModel = XGBClassifier()
                             

    x_train, x_test, y_train, y_test =train_test_split(X_out, y_out,test_size=0.1,random_state=0)
    st.write()
        
    XgbModel.fit(x_train, y_train)
    ypred = XgbModel.predict(x_test)
    
    
    xg_acc = accuracy_score(y_test, ypred)
    st.write("The accuracy before feature engineering is 74.4%")
    st.write(f"The accuracy after feature engineering is 87%")
    
    conf = confusion_matrix(y_test,ypred)
    st.plotly_chart(px.imshow(conf,text_auto=True))
    
    
    
    feat_labels = x2.columns
    model = RandomForestClassifier(n_estimators=100, criterion='gini', min_samples_split=5, min_samples_leaf=2, max_features='auto', bootstrap=True, n_jobs=-1, random_state=42)
    model.fit(x_train,y_train)
    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1]
    
    fe_fig = plt.figure(figsize=(6, 4))
    plt.title('Feature importances')
    plt.bar(range(x_train.shape[1]), importances[indices], align='center')
    plt.xticks(range(x_train.shape[1]),  feat_labels[indices], rotation=90)
    plt.xlim([-1, x_train.shape[1]])
    plt.tight_layout()
    st.pyplot()
    
    st.write(" From the above plot we can see the feature importances arranged from highest to "
             " lowest. My intuiation to remove systolic column from the data was right. But, however low the effect of a attribute is on the data "
             " it is not right to drop them. The co-efficients still have some value in the context of machine learning.")



    


elif (d =='PCA'):
    
    st.markdown("1. Let's see the principle component analysis. " )
    
    st.markdown("let's see the PCA after standard scaling is applied")
    
    
    
    
    
    scaler = StandardScaler()
    X2= scaler.fit_transform(x2)
    
    pca = PCA()
    pca.fit_transform(X2)
    explained_variance=pca.explained_variance_ratio_
    with plt.style.context('dark_background'):
        pca_fig = plt.figure(figsize=(6, 4))
    
        pca_fig = plt.bar(range(11), explained_variance, alpha=0.5, align='center',
                label='individual explained variance')
        plt.ylabel('Explained variance ratio')
        plt.xlabel('Principal Components')
        plt.legend(loc='best')
        plt.tight_layout()
        st.set_option('deprecation.showPyplotGlobalUse', False)
        st.pyplot()
        
        st.subheader("Observations")
        st.write("1.Two components explains more than 50% of the variance of data.  \n"
                 "2. Four components can explain 85% of the data")
        
        explained_variance = plt.figure(figsize=(6, 4))
        
        exp_var_cumul = np.cumsum(pca.explained_variance_ratio_)
        
        explained_variance = px.area( x=range(1, exp_var_cumul.shape[0] + 1),y=exp_var_cumul,
    labels={"x": "# Components", "y": "Explained Variance"},color_discrete_sequence=["dodgerblue"])
        st.plotly_chart(explained_variance)
        


elif(d=='HyperParameter Tuning'):
    
    ht_radio = st.radio(label='select one',options=('Random Forest','SVM','XGboost'))
    
    if(ht_radio=="Random Forest"): 
    
        scaler = StandardScaler()
            
        X_out=scaler.fit_transform(x2)
                 
        x_train, x_test, y_train, y_test =train_test_split(x2, y,test_size=0.1,random_state=0)
        
        dt = GridSearchCV(RandomForestClassifier(random_state=50),{'criterion':['gini', 'entropy', 'log_loss'],'max_depth':[1,5,10]},cv=5,return_train_score = False)
        dt.fit(x_train,y_train)
        st.write(dt.best_params_)
        st.write(dt.best_score_)
        st.write("Hyperparameter tuning consists of finding a set of optimal hyperparameter values "
                 " for a learning algorithm while applying this optimized algorithm to any data set. That combination of hyperparameters maximizes the model’s performance, "
                 "minimizing a predefined loss function to produce better results with fewer errors.")
        st.write("Important parameters for Random Forest are:")
        st.markdown("1.criterion")
        st.markdown("Max depth")
        
    
    elif(ht_radio=="SVM"):
        x_train, x_test, y_train, y_test =train_test_split(x2, y,test_size=0.1,random_state=0)
        dt1 = GridSearchCV(estimator=SVC(),param_grid={'C': [5, 10], 'kernel': ('linear', 'rbf')})
        dt1.fit(x_train,y_train)
        st.write(dt.best_params_)
        st.write(dt.best_score_)



elif (d =='Models'): 
    
    model_options = st.radio(
        "Select the type of disease plot:",
        ('XGboost','SVM','Random Forest','K-Means'))
    
    if (model_options=='XGboost'):
        
        scaler = StandardScaler()
            
        X_out=scaler.fit_transform(x2)

        XgbModel = XGBClassifier()
                                 

        x_train, x_test, y_train, y_test =train_test_split(x2, y,test_size=0.1,random_state=0)
        st.write()
            
        XgbModel.fit(x_train, y_train)
        ypred = XgbModel.predict(x_test)
        
        
        xg_acc = accuracy_score(y_test, ypred)
        
        st.write(f"The accuracy of this model is {xg_acc}")
        
        conf = confusion_matrix(y_test,ypred)
        st.plotly_chart(px.imshow(conf,text_auto=True))
        
        
        st.write("Let's try out the predictions.")
        
        gender_svm = st.select_slider("Enter the gender",options=[0,1],value=0)
        age_svm = st.slider("Enter the age",min_value=21, max_value=64,value=44)
        
        height_svm = st.number_input("Enter Height",130.0,185.0,value=165.0,step=0.1)
        
        weight_svm = st.number_input("Enter weight",50.0,110.0,value=55.8,step=0.1)
        
        bodyfat_p_svm = st.number_input("Enter the body fat percent",min_value=3.0, max_value=78.0,value=15.7,step=0.1)
        distolic_svm = st.slider("Enter the diastolic value ",min_value=0.0, max_value=156.0,value=77.0,step=0.5)
        
        systolic_svm = st.number_input("Enter the systolic value ",min_value=0.0, max_value=201.0,value=126.0,step=0.5)
        grip_svm = st.number_input("Enter the grip force value ",min_value=0.0, max_value=70.0,value=36.4,step=0.1)
        
        sitbend_svm = st.number_input("Enter Sit and bend forward stretch",min_value=0.0,max_value=16.30,value=9.3,step=0.1)
        sit_up_svm = st.number_input("Enter sit up count",min_value= 0.0,max_value=60.0,value=53.0,step=0.5)
        
        broadjump_svm = st.slider("Enter the broad jump value ",min_value=0, max_value=303,value=229)
        
        X_d = pd.DataFrame({"age": age_svm,"gender":gender_svm, "height_cm":height_svm, "weight_kg":weight_svm,"body fat_%":bodyfat_p_svm,
                                "diastolic":distolic_svm,"systolic":systolic_svm, "gripForce":grip_svm,"sit and bend forward_cm":sitbend_svm,
                                "sit-ups counts":sit_up_svm,"broad jump_cm": broadjump_svm},index=[0])
                     
            
        dummy = x2
        dummy = dummy.append(X_d,ignore_index= True)
         
        dummy = pd.get_dummies(dummy,drop_first=True)
        dummy= scaler.fit_transform(dummy)
            
            
        prediction =  XgbModel.predict(X_d)
        if (prediction==0):
            st.write("You are Fit💪")
            
        else:
            st.write("Sorry, You are not fit !😌")
        
        labels = ["fit","UnFit"]   
        
        clf_report = classification_report(y_test,
                                   ypred,
                                   labels=[0,1],
                                   target_names=labels,
                                   output_dict=True)
        
        fig = plt.figure(figsize = (8,5))
        sns.heatmap(pd.DataFrame(clf_report).iloc[:-1, :].T, annot=True)
        plt.legend()
        title = "Classification Report"
        plt.title(title)
        st.pyplot(fig)
            
    elif(model_options=='SVM'):
        x_train, x_test, y_train, y_test =train_test_split(x2, y,test_size=0.1,random_state=0)
        
        
        
        svm_model = SVC(gamma=1e-4 ,C=1000,random_state=42)
        svm_model = svm_model.fit(x_train, y_train)
        
        Y_pred = svm_model.predict(x_test)
        svm_acc = accuracy_score(y_test, Y_pred)
        st.write(svm_acc)
        
        conf = confusion_matrix(y_test,Y_pred)
        st.plotly_chart(px.imshow(conf,text_auto=True))
        
        st.write("Let's try out the predictions.")
        
        gender_svm = st.select_slider("Enter the gender",options=[0,1],value=1)
        age_svm = st.slider("Enter the age",min_value=21, max_value=64,value=25)
        
        height_svm = st.number_input("Enter Height",130.0,185.0,value=165.0,step=0.1)
        
        weight_svm = st.number_input("Enter weight",50.0,110.0,value=55.8,step=0.1)
        
        bodyfat_p_svm = st.number_input("Enter the body fat percent",min_value=3.0, max_value=78.0,value=15.7,step=0.1)
        distolic_svm = st.slider("Enter the diastolic value ",min_value=0.0, max_value=156.0,value=77.0,step=0.5)
        
        systolic_svm = st.number_input("Enter the systolic value ",min_value=0.0, max_value=201.0,value=126.0,step=0.5)
        grip_svm = st.number_input("Enter the grip force value ",min_value=0.0, max_value=70.0,value=36.4,step=0.1)
        
        sitbend_svm = st.number_input("Enter Sit and bend forward stretch",min_value=0.0,max_value=16.30,value=9.3,step=0.1)
        sit_up_svm = st.number_input("Enter sit up count",min_value= 0.0,max_value=60.0,value=53.0,step=0.5)
        
        broadjump_svm = st.slider("Enter the broad jump value ",min_value=0, max_value=303,value=229)
        
        X_d = pd.DataFrame({"age": age_svm,"gender":gender_svm, "height_cm":height_svm, "weight_kg":weight_svm,"body fat_%":bodyfat_p_svm,
                                "diastolic":distolic_svm,"systolic":systolic_svm, "gripForce":grip_svm,"sit and bend forward_cm":sitbend_svm,
                                "sit-ups counts":sit_up_svm,"broad jump_cm": broadjump_svm},index=[0])
                     
            
        
            
            
        prediction =  svm_model.predict(X_d)
        if (prediction==0):
            st.write("You are Fit💪")
            
        else:
            st.write("Sorry, You are not fit !😌")
            
        labels = ["fit","UnFit"]    
        clf_report = classification_report(y_test,
                                   Y_pred,
                                   labels=[0,1],
                                   target_names=labels,
                                   output_dict=True)
        
        fig = plt.figure(figsize = (8,5))
        sns.heatmap(pd.DataFrame(clf_report).iloc[:-1, :].T, annot=True)
        plt.legend()
        title = "Classification Report"
        plt.title(title)
        st.pyplot(fig)
            
            
            
    elif(model_options=='Random Forest'):
        
        x_train, x_test, y_train, y_test =train_test_split(x2, y,test_size=0.1,random_state=0)
        
        model = RandomForestClassifier(n_estimators=100, criterion='gini', min_samples_split=5, min_samples_leaf=2, max_features='auto', bootstrap=True, n_jobs=-1, random_state=42)
        model.fit(x_train,y_train)
        y_pred = model.predict(x_test)
        rf_acc = accuracy_score(y_test, y_pred)
        
        st.write(f"The accuracy of Random forest is {rf_acc}")
        
        conf_rf = confusion_matrix(y_test, y_pred)
        st.plotly_chart(px.imshow(conf_rf,text_auto=True))
        
        st.write("Let's try out the predictions.")
        
        gender_svm = st.select_slider("Enter the gender",options=[0,1],value=1)
        age_svm = st.slider("Enter the age",min_value=21, max_value=64,value=54)
        
        height_svm = st.number_input("Enter Height",130.0,185.0,value=165.0,step=0.1)
        
        weight_svm = st.number_input("Enter weight",50.0,110.0,value=55.8,step=0.1)
        
        bodyfat_p_svm = st.number_input("Enter the body fat percent",min_value=3.0, max_value=78.0,value=15.7,step=0.1)
        distolic_svm = st.slider("Enter the diastolic value ",min_value=0.0, max_value=156.0,value=77.0,step=0.5)
        
        systolic_svm = st.number_input("Enter the systolic value ",min_value=0.0, max_value=201.0,value=126.0,step=0.5)
        grip_svm = st.number_input("Enter the grip force value ",min_value=0.0, max_value=70.0,value=36.4,step=0.1)
        
        sitbend_svm = st.number_input("Enter Sit and bend forward stretch",min_value=0.0,max_value=16.30,value=9.3,step=0.1)
        sit_up_svm = st.number_input("Enter sit up count",min_value= 0.0,max_value=60.0,value=43.0,step=0.5)
        
        broadjump_svm = st.slider("Enter the broad jump value ",min_value=0, max_value=303,value=229)
        
        X_d = pd.DataFrame({"age": age_svm,"gender":gender_svm, "height_cm":height_svm, "weight_kg":weight_svm,"body fat_%":bodyfat_p_svm,
                                "diastolic":distolic_svm,"systolic":systolic_svm, "gripForce":grip_svm,"sit and bend forward_cm":sitbend_svm,
                                "sit-ups counts":sit_up_svm,"broad jump_cm": broadjump_svm},index=[0])
                     
            
        
            
            
        prediction =  model.predict(X_d)
        if (prediction==0):
            st.write("You are Fit💪")
            
        else:
            st.write("Sorry, You are not fit !😌")
            
        clf_report = classification_report(y_test,
                                   y_pred,
                                   labels=[0,1],
                                   target_names=labels,
                                   output_dict=True)
        
        fig = plt.figure(figsize = (8,5))
        sns.heatmap(pd.DataFrame(clf_report).iloc[:-1, :].T, annot=True)
        plt.legend()
        title = "Classification Report"
        plt.title(title)
        st.pyplot(fig)
        
    elif(model_options=='K-Means'):
        
        Model = KMeans(n_clusters=2 ,init="k-means++", max_iter=1000)
        
        Model.fit_predict(data_outlier[['weight_kg','height_cm']])
        
        x_km = data_outlier.copy()
        
        x_km = x_km.drop(columns=['class'])
        
        
        x_km['Clusters'] = Model.labels_
        st.write(x_km)
        st.subheader("Before Clustering")
        fig_km = px.scatter(data_outlier,x="weight_kg", y="height_cm", color= 'cls_cat')
        st.plotly_chart(fig_km)
        st.subheader("After Clustering")
        fig_kmn = px.scatter(x_km,x="weight_kg", y="height_cm", color= 'Clusters')
        st.plotly_chart(fig_kmn)
        

        
        
            
    
            
            
            
        
        
        
            
        
        
        
        
    
    
    
    
    
    
    
    
    
    
    




