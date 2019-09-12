import numpy as np
import matplotlib.pyplot as plt



class Waitlist:
    def __init__(self, num_people=10, length_book=400, speed=200, words_page=500):

        self.num_people = self.number_of_people(num_people)
        self.length_book = self.length_of_book(length_book)
        self.speed = self.reading_speed(speed)  # WPM
        self.words_page = self.words_per_page(words_page)
        self.MINUTES_PER_DAY = 1440

    def number_of_people(self, cen, dist=np.random.randint, **kwargs):
        return dist(cen, **kwargs)

    def length_of_book(self, cen, dist=np.random.poisson, **kwargs):
        return dist(cen, **kwargs)

    def reading_speed(self, cen, width=50, dist=np.random.normal, **kwargs):
        return dist(cen, width, **kwargs)

    def words_per_page(self, cen, width=100, dist=np.random.normal, **kwargs):
        return dist(cen, width, **kwargs)

    def model(self):
        return self.num_people * abs(self.length_book * self.words_page / self.speed) / self.MINUTES_PER_DAY

plt.cla()
bins = np.linspace(-1, 30, 31)
draws = [Waitlist().model() for i in range(10000)]
plt.hist(draws, bins=bins, histtype='step', density=True)
plt.xlabel('Number of Days on Waitlist', fontsize=26)
plt.tight_layout()
plt.show()