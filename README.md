# Knowledge-Fusion-for-Occupancy-Estimation
The data and code in this repositiory was the basis to the paper "Lu C. Enhancing real-time nonintrusive occupancy estimation in buildings via knowledge fusion network[J]. Energy and Buildings, 2024, 303: 113812."

* **Data**:
  The dataset is from the ASHRAE Global Building Occupant Behavior Database. The climatic condition is Mediterranean hot-summer climate. The dataset contains the number of occupants in the office, as well as the indoor environment (indoor air temperature, indoor air relative humidity, CO2, and TVOC), the electricity usage, the occupant actions (window state, door state, and AC operations), and the contextual information (Hour of day). The initial resolution of the dataset is one minute. The dataset is from 8 am to 9 pm on the working days (from May 13th to July 8th, 2016), which also adds the selected handcrafted features.
![Data2](https://github.com/user-attachments/assets/d8c99538-c7f9-4012-99e8-1ec050d603b7)

* **KFN**
  The main part of the proposed knowledge fusion network for real-time nonintrusive occupancy estimation in buildings. The code is taking the average merging as an example.

To be able to run all the models you will need (along with other standard python packages):
  * sckikit-learn
  * keras
  * tensorflow
