# <center>__Model Final Results__</center>

### <center>__Overall Results:__</center>
#### <center>__Top Result:__ Random Forest</center>

<table>

|Model|F1 Score|AUC|Pipeline|SMOTE|Feature Selection|
|----:|:------:|:-:|:------:|:---:|:-------:|
|__Logistic Regression__|0.6603|0.8208|3.1|Yes|Yes|
|__ADA Boost__|0.6593|0.8352|3.2|Yes|Yes|
|__Random Forest__|0.6636|0.8261|2|Yes|Yes|
|__SVM__|0.6446|0.8553|1|Yes|Yes|
|__Decision Tree__|0.6474|0.8356|1, 2, 3.2|Yes|Yes|
|__XG Boost__|0.6537|0.8212|2|Yes|Yes|

</table>
    
### __Individual Results:__
#### __Model 1: Logistic Regression:__

__Top Result:__ Feature Selection with SMOTE Pipeline 3.1

<table>
<tr><th>Without SMOTE:</th><th>With SMOTE:</th></tr>
<tr><td>

|Pipeline|F1 Score|AUC|
|:------:|--------|---|
|__1__|0.6159|0.7635|
|__2__|0.60497|0.7585|
|__3.1__|0.6166|0.7636|
|__3.2__|0.5921|0.731|

</td><td>

|Pipeline|F1 Score|AUC|
|:------:|--------|---|
|__1__|0.6141|0.8291|
|__2__|0.6283|0.8323|
|__3.1__|0.6125|0.8278|
|__3.2__|0.6655|0.8096|

</td></tr> </table>

<table>
<tr><th>Feature Selection Without SMOTE:</th><th>Feature Selection With SMOTE:</th></tr>
<tr><td>

|Pipeline|F1 Score|AUC|
|:------:|--------|---|
|__1__|0.6265|0.7826|
|__2__|0.6320|0.7855|
|__3.1__|0.6297|0.7585|
|__3.2__|0.|0.|
    
</td><td>

|Pipeline|F1 Score|AUC|
|:------:|--------|---|
|__1__|0.6381|0.8318|
|__2__|0.6393|0.8329|
|__3.1__|0.6603|0.8208|
|__3.2__|0.|0.|

</td></tr> </table>
        
#### __Model 2: ADA Boost:__

__Top Result:__ Feature Selection with SMOTE Pipeline 3.2

<table>
<tr><th>Without SMOTE:</th><th>With SMOTE:</th></tr>
<tr><td>

|Pipeline|F1 Score|AUC|
|:------:|--------|---|
|__1__|0.6005|0.7575|
|__2__|0.5991|0.7572|
|__3.1__|0.6031|0.7596|
|__3.2__|0.6038|0.7598|

</td><td>

|Pipeline|F1 Score|AUC|
|:------:|--------|---|
|__1__|0.6529|0.8275|
|__2__|0.6454|0.8268|
|__3.1__|0.6416|0.8184|
|__3.2__|0.6475|0.8296|

</td></tr> </table>

<table>
<tr><th>Feature Selection Without SMOTE:</th><th>Feature Selection With SMOTE:</th></tr>
<tr><td>

|Pipeline|F1 Score|AUC|
|:------:|--------|---|
|__1__|0.6116|0.7557|
|__2__|0.6061|0.7532|
|__3.1__|0.5958|0.7458|
|__3.2__|0.6104|0.7554|

</td><td>

|Pipeline|F1 Score|AUC|
|:------:|--------|---|
|__1__|0.6542|0.8292|
|__2__|0.6556|0.8286|
|__3.1__|0.6561|0.8282|
|__3.2__|0.6593|0.8352|

</td></tr> </table>

#### __Model 3: Random Forest:__

__Top Result:__ Feature Selection with SMOTE Pipeline 2 

<table>
<tr><th>Without SMOTE:</th><th>With SMOTE:</th></tr>
<tr><td>

|Pipeline|F1 Score|AUC|
|:------:|--------|---|
|__1__|0.6081|0.7405|
|__2__|0.6027|0.74|
|__3.1__|0.6072|0.74|
|__3.2__|0.5921|0.731|

</td><td>

|Pipeline|F1 Score|AUC|
|:------:|--------|---|
|__1__|0.6557|0.7986|
|__2__|0.6644|0.8132|
|__3.1__|0.6490|0.7967|
|__3.2__|0.6655|0.8096|

</td></tr> </table>

<table>
<tr><th>Feature Selection Without SMOTE:</th><th>Feature Selection With SMOTE:</th></tr>
<tr><td>

|Pipeline|F1 Score|AUC|
|:------:|--------|---|
|__1__|0.6221|0.7580|
|__2__|0.6189|0.7520|
|__3.1__|0.6297|0.7584|
|__3.2__|0.6214|0.7527|

</td><td>

|Pipeline|F1 Score|AUC|
|:------:|--------|---|
|__1__|0.6639|0.8119|
|__2__|0.6636|0.8261|
|__3.1__|0.6650|0.8178|
|__3.2__|0.6651|0.8211|

</td></tr> </table>

#### __Model 4: SVM:__

__Top Result:__ Feature Selection with SMOTE Pipeline 1

<table>
<tr><th>Without SMOTE:</th><th>With SMOTE:</th></tr>
<tr><td>

|Pipeline|F1 Score|AUC|
|:------:|--------|---|
|__1__|0.6259|0.7613|
|__2__|0.6178|0.7690|
|__3.1__|0.6273|0.7627|
|__3.2__|0.0|0.5000|

</td><td>

|Pipeline|F1 Score|AUC|
|:------:|--------|---|
|__1__|0.6517|0.8459|
|__2__|0.6345|0.8274|
|__3.1__|0.6494|0.8451|
|__3.2__|0.3581|0.6282|

</td></tr> </table>

<table>
<tr><th>Feature Selection Without SMOTE:</th><th>Feature Selection With SMOTE:</th></tr>
<tr><td>

|Pipeline|F1 Score|AUC|
|:------:|--------|---|
|__1__|0.6288|0.7647|
|__2__|0.6172|0.7668|
|__3.1__|0.616|0.7555|
|__3.2__|0.0|0.5000|

</td><td>

|Pipeline|F1 Score|AUC|
|:------:|--------|---|
|__1__|0.6446|0.8553|
|__2__|0.6283|0.8342|
|__3.1__|0.6349|0.8479|
|__3.2__|0.3713|0.6643|

</td></tr> </table>

#### __Model 5: Decision Tree:__

__Top Result:__ Feature Selection with SMOTE Pipelines 1, 2, 3.2

<table>
<tr><th>Without SMOTE:</th><th>With SMOTE:</th></tr>
<tr><td>

|Pipeline|F1 Score|AUC|
|:------:|--------|---|
|__1__|0.5288|0.7228|
|__2__|0.5554|0.7423|
|__3.1__|0.5536|0.7366|
|__3.2__|0.5536|0.7403|

</td><td>

|Pipeline|F1 Score|AUC|
|:------:|--------|---|
|__1__|0.5634|0.7558|
|__2__|0.5627|0.7560|
|__3.1__|0.5294|0.7304|
|__3.2__|0.5678|0.7616|

</td></tr> </table>

<table>
<tr><th>Feature Selection Without SMOTE:</th><th>Feature Selection With SMOTE:</th></tr>
<tr><td>

|Pipeline|F1 Score|AUC|
|:------:|--------|---|
|__1__|0.5879|0.7477|
|__2__|0.5879|0.7477|
|__3.1__|0.5879|0.7477|
|__3.2__|0.5879|0.7477|

</td><td>

|Pipeline|F1 Score|AUC|
|:------:|--------|---|
|__1__|0.6474|0.8356|
|__2__|0.6474|0.8356|
|__3.1__|0.5751|0.8291|
|__3.2__|0.6474|0.8356|

</td></tr> </table>

#### __Model 6: XG Boost:__

__Top Result:__ Feature Selection with SMOTE Pipeline 2

<table>
<tr><th>Without SMOTE:</th><th>With SMOTE:</th></tr>
<tr><td>

|Pipeline|F1 Score|AUC|
|:------:|--------|---|
|__1__|0.5793|0.7348|
|__2__|0.597|0.7450|
|__3.1__|0.5789|0.7339|
|__3.2__|0.5924|0.7414|

</td><td>

|Pipeline|F1 Score|AUC|
|:------:|--------|---|
|__1__|0.6366|0.8032|
|__2__|0.6456|0.8067|
|__3.1__|0.6422|0.8048|
|__3.2__|0.6488|0.8067|

</td></tr> </table>

<table>
<tr><th>Feature Selection Without SMOTE:</th><th>Feature Selection With SMOTE:</th></tr>
<tr><td>

|Pipeline|F1 Score|AUC|
|:------:|--------|---|
|__1__|-|-|
|__2__|-|-|
|__3.1__|-|-|
|__3.2__|-|-|

</td><td>

|Pipeline|F1 Score|AUC|
|:------:|--------|---|
|__1__|0.6427|0.8169|
|__2__|0.6537|0.8212|
|__3.1__|0.6402|0.8145|
|__3.2__|0.6513|0.8204|

</td></tr> </table>