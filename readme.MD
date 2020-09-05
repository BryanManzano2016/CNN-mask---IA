# Sistema de detección de mascarillas

A continuación se procedera a explicar como instalar y ejecutar el sofware.
## Instalación
Para la ejecucion del programa es necesaria la version 3.7 de Python

Con el administrador de paquetes pip

Instalar tensorflow-cpu y matplotlib
```
pip install tensorflow-cpu
pip install matplotlib
```

El resto de librerias vienen con el sistema, por tanto no hay que instalarlas con pip. Estas son json, redirect_stdout, datetime y os

## Resultado de ejecutar
Se genera una carpeta con formato model_project_YYYYMMDD, en la cual se guarda lo siguiente:

Alli encontrara:
* model.txt : resultado de ejecutar la función summary() de tensorflow
* model.h5 : modelo resultado de la ejecución 
* params.txt : diccionario en formato json con el información de la variable params del proyecto.py
* pesos.ht : pesos resultado de la ejecución de tensorflow
* plot.png : gráfico obtenido de matplotlib


 