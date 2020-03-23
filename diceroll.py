import numpy as np
from matplotlib import use
use('tkAgg')
import matplotlib.pyplot as plt

low = 1
high = int(input('Number of sides on dice: '))
num = int(input('Number of dice: '))
rolls = int(input('Number of rolls: '))

f, ax = plt.subplots(1,1, figsize=(6,6))
A = np.random.randint(low, high+1, rolls)
for i in range(1, num):
	A += np.random.randint(low, high+1, rolls)

ax.hist(A, density=True)
ax.set_xlabel('Sum of dice', fontsize=20)
ax.text(0.1, 0.9, s=str(high)+' sided dice', fontsize=14, color='k', transform=ax.transAxes)
ax.text(0.1, 0.85, s=str(num)+' dice', fontsize=14, color='k', transform=ax.transAxes)
ax.text(0.1, 0.8, s=str(rolls)+' rolls', fontsize=14, color='k', transform=ax.transAxes)

plt.tight_layout()
