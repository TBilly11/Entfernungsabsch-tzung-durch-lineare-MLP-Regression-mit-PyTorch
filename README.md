# Entfernungsabsch-tzung-durch-lineare-MLP-Regression-mit-PyTorch

<h1> 1. Einleitung und Zielsetzung</h1>

Im Rahmen der Projektarbeit soll im Bereich Computer Vision ein Distance Estimation Modell erstellt, erprobt und in Echtzeit eingesetzt werden. Das Ziel ist es, ein System zu entwickeln, das in der Lage ist, Personen auf einer Bühne mithilfe einer Kamera zu verfolgen und deren Entfernung zu schätzen. Dabei sollen vorhandene Datensätze ermittelt und ein Regressionsmodell zur Distanzschätzung erstellt werden, wobei PyTorch als Framework zur Implementierung des Modells genutzt wird.
Um ein robustes und effektives Modell zur Distanzschätzung zu entwickeln, wird eine detaillierte Untersuchung der verfügbaren Datensätze und deren Eignung für diese spezifische Anwendung durchgeführt. Basierend auf dieser Analyse wird ein Regressionsmodell mit Hilfe von Pytorch entwickelt und optimiert. Dabei werden verschiedene Netzwerkarchitekturen und Trainingsstrategien getestet, um die beste Leistung zu erzielen. Das Endziel ist die Entwicklung eines Modells, das in der Lage ist, die Entfernung der Personen zur Kamera in Echtzeit präzise zu schätzen und somit die Grundlage für fortschrittliche Personenverfolgungssysteme auf Bühnen und anderen Anwendungen zu bilden.

<h1> 2. Hintergrund </h1>

Die präzise Schätzung von Entfernungen ist eine wesentliche Herausforderung in vielen Anwendungen der Computer Vision, insbesondere in Echtzeitsystemen wie der Personenverfolgung auf einer Bühne. Hierbei sollen die gemessenen Bilddaten wie Höhe, Breite, Zoom und Fokus verwendet werden, um die Entfernung der Personen zur Kamera zuverlässig zu bestimmen. Dies ermöglicht es, die Position und Bewegung der Personen auf der Bühne genau zu verfolgen, was in verschiedenen Bereichen wie der Bühnenproduktion, Sicherheitssystemen und interaktiven Anwendungen von großer Bedeutung ist.
Eine umfassende Untersuchung der bestehenden Literatur und Technologien im Bereich der Distance Estimation und Personenverfolgung wird durchgeführt. Ziel ist es, die aktuellen Methoden und Herausforderungen zu verstehen.

<h1> 2.1. Wichtigkeit eines festen Seeds in der Datenanalyse </h1>

In der linearen MLP-Regression (Multi-Layer Perceptron) wird der Seed verwendet, um den Zufallszahlengenerator zu initialisieren. Dieser Zufallszahlengenerator wird in verschiedenen Schritten des Trainingsprozesses eingesetzt, z.B. beim:

Initialisieren der Modellgewichte:  
Die Gewichte in einem neuronalen Netzwerk werden normalerweise zufällig initialisiert, bevor das Training beginnt.

Aufteilen der Daten:  
Daten werden oft in Trainings-, Validierungs- und Test-Sets aufgeteilt, was oft zufällig geschieht.

Ein fester Seed sorgt dafür, dass der Zufallszahlengenerator jedes Mal die gleichen "zufälligen" Werte erzeugt, was die folgenden Vorteile bietet:

**Reproduzierbarkeit**:  
Durch das Setzen eines festen Seeds kann man sicherstellen, dass Experimente wiederholbar sind.

**Vergleichbarkeit**:  
Wenn man verschiedene Modelle, Algorithmen oder Konfigurationen vergleichen möchte, ist es wichtig, dass diese unter denselben Bedingungen getestet werden.

<h1> 2.2. Modell Optimisation </h1>

Beim Einsatz von MLP-Regression mit PyTorch gibt es verschiedene Möglichkeiten, die Modelle zu verbessern und damit genauere Vorhersagen zu erzielen. Einige Methoden zur Optimierung des Modells, um den MSE (Mean Squared Error) zu reduzieren, sind unter anderem: 

- **Erhöhen der Modellkomplexität**  
- **Verwenden von Regularisierungstechniken**

Während des Trainings wurden verschiedene Modelle getestet: ein kleines, ein mittleres und ein großes Modell. Um eine faire und effiziente Vergleichbarkeit sicherzustellen, wurde ein fester Seed verwendet. Zudem kamen Regularisierungstechniken wie Dropout zum Einsatz, um Overfitting zu reduzieren und die Generalisierungsfähigkeit der Modelle zu verbessern.

<br />
Kleines Modell ohne Dropout:  <br/>
<img src="https://i.imgur.com/lzUG1l5.png" style="max-width:100%; height:auto; display:block; margin:auto;" alt="Disk Sanitization Steps"/>
<br />

Mittleres Modell ohne Dropout:  <br/>
<img src="https://i.imgur.com/cmEXAH6.png" style="max-width:100%; height:auto; display:block; margin:auto;" alt="Disk Sanitization Steps"/>
<br />

Großes Modell ohne Dropout:  <br/>
<img src="https://i.imgur.com/qznarOL.png" style="max-width:100%; height:auto; display:block; margin:auto;" alt="Disk Sanitization Steps"/>
<br />

<img src="https://i.imgur.com/bzXMC9Z.png" style="max-width:100%; height:auto; display:block; margin:auto;" alt="Disk Sanitization Steps"/>
<br />

<img src="https://i.imgur.com/Lquj52l.png" style="max-width:100%; height:auto; display:block; margin:auto;" alt="Disk Sanitization Steps"/>
<br />

<img src="https://i.imgur.com/taAFL42.png" style="max-width:100%; height:auto; display:block; margin:auto;" alt="Disk Sanitization Steps"/>
<br />

<img src="https://i.imgur.com/qdfqJL0.png" style="max-width:100%; height:auto; display:block; margin:auto;" alt="Disk Sanitization Steps"/>
<br />

<img src="https://i.imgur.com/qdfqJL0.png" style="max-width:100%; height:auto; display:block; margin:auto;" alt="Disk Sanitization Steps"/>
<br />
