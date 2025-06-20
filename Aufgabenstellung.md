[ACHTUNG! HABE DEN TEXT NICHT ÜBERPRÜFT]

Hier ist der Text aus dem Bild in 1:1-Format:

---

### **3 Projekt**

In diesem Projekt geht es darum, mit Hilfe von Monte-Carlo-Simulationen eine optimale Auslegung eines optischen Messsystems zu gewinnen. Dies ist im Wesentlichen eine Erweiterung und spezielle Anwendung der Erkenntnisse aus der Vorlesung zur Positionsbestimmung.

#### **3.1 Theoretische Vorarbeit zur Multilateration**

→ Wie in der Vorlesung betrachten wir die Positionsbestimmung in einer 2D-Ebene. Sie haben hierzu Abstandsmessergeräte zur Verfügung, wie Sie sie für die Lateration benötigen. In unserem Fall kann jedes Abstandsmessergerät in 360° aussenden und empfangen.

→ Stellen Sie hierzu alle mathematischen Gleichungen auf, die es Ihnen erlauben, an gegebenen Positionen \( z_1, z_2, \ldots, z_n \), welche die Abstände \( r_1, r_2, \ldots, r_n \) zu einem Objekt \( \vec{y} \) messen, eine optimale Schätzung des Ortes \( \vec{y} \) zu gewinnen. Gehen Sie davon aus, dass die Entfernungsmaßwerte \( r_1, r_2, \ldots, r_n \) einem Messrauschen unterliegen, sodass Sie alle Messungen in der Positionsbestimmung berücksichtigen müssen, um einen optimalen Fehlerausgleich zu erhalten.

**Hinweis:** Verwenden Sie für diese Aufgabe die quadratische Zielfunktion, um den Fehlerausgleich durchzuführen. Dies ist auch ein anderer Ansatz als die Schnittpunktbestimmung von zwei Kreisen, wie in der Vorlesung.

→ Dokumentieren Sie diese Formeln und Ihre mathematische bzw. algorithmische Lösungsstrategie möglichst übersichtlich und klar.

#### **3.2 Multilateration in gegebener Geometrie**

→ Nun sollen Sie ein Setup der Messung für eine konkrete praktische Anwendung vorschlagen.

→ Sie haben 3 Abstandsmessergeräte zur Verfügung. Das Rauschen jedes Abstandsmessergeräts nimmt mit der gemessenen Entfernung \( r \) zu. Für die 3 Abstandsmessergeräte gelten folgende Kennlinien \( \sigma(r) \) in [mm]:

| Messgerät 1 | \( \sigma(r) = 2.5 + 0.0010 \cdot r \) |
| Messgerät 2 | \( \sigma(r) = 5.0 + 0.0005 \cdot r \) |
| Messgerät 3 | \( \sigma(r) = 0.5 + 0.0020 \cdot r \) |

mit \( \sigma(r) \) der Standardabweichung eines normalverteilten Messfehlers.

→ Weiterhin geht es um die Ausmessung von Positionen in einem Raum folgenden Grundrisses:

![Ellipse](https://i.imgur.com/ellipse.png)

d.h. ein elliptischer Raum mit vertikaler Halbachse \( A \) und horizontaler Halbachse \( B \), bei dem ein Segment des Winkels \( \alpha \) fehlt (Spitze ist im Mittelpunkt der Ellipse). Sie dürfen die Abstandsmessergeräte mit den Positionen \( \vec{z}_1, \vec{z}_2, \vec{z}_3 \) nur an der Umrandung, d.h. an den Wänden, des Raums anbringen.

→ Für Ihre Projektgruppe gelten folgende Maße:

| Gruppen-Nr. | 1       | 2       | 3       | 4       | 5       | 6       | 7       |
|-------------|---------|---------|---------|---------|---------|---------|---------|
| \( A \)     | 11000 mm | 10000 mm | 9000 mm | 8000 mm | 7000 mm | 6000 mm | 5000 mm |
| \( B \)     | 6000 mm | 7000 mm | 8000 mm | 9000 mm | 10000 mm | 11000 mm | 12000 mm |
| \( \alpha \) | 50°    | 55°    | 60°    | 65°    | 70°    | 75°    | 80°    |

Modellbildung und Simulation, Sommersemester 2025. © Prof. Dr. Wolfgang Högele

→ Die Abstandsmessergeräte können insbesondere nicht durch eine Mauer hindurch messen, d.h. die Mauern können ein Messgerät abschatten. Daher an manchen möglichen Objektpositionen können weniger als drei Abstandsmessergeräte das Objekt sehen, weil manche Messgeräte abgeschirmt werden.

→ Finden Sie für jedes Messgerät \( i \) (1, 2, 3) die dazugehörige Position \( \vec{z}_i \), sodass im gesamten Raum die beste Messperfomanz erreicht wird. Überlegen Sie hierzu den Leser, dass Sie tatsächlich den gesamten Raum untersuchen (beispielsweise eine strikte, feine Abtastung des Raums mit mind. 500 möglichen Objektpositionen) und eine Monte-Carlo-Simulation für jeden einzelnen Objekttort. Bestimmen Sie das Ergebnis des RMS-Wert der Positionsbestimmung an diesem Objekttort. Berechnen Sie das 95%-Quantil der RMS-Werte aller Objekte für die Güte des Mess-Setups im gesamten Raum. Stellen Sie mindestens zwei unvorteilhafte Mess-Setups gegenüber dem besten Setup, das Sie gefunden haben, graphisch ansprechend dar.

→ Die Beschreibung beinhaltet auch Details zu Ihrer Monte-Carlo-Simulation, deren Parameterwahl und deren Auswertung. Ebenso müssen Sie alle Ergebnisse graphisch intuitiv verständlich darstellen, aus denen man erkennen wo im Raum wie gut gemessen wird.

---

Das ist der vollständige Text aus dem Bild.