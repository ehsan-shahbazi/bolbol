from django.db import models
import pandas as pd
from django.utils import timezone
from django.http import request as django_req
from bs4 import BeautifulSoup
from requests import Session
import trendln
import matplotlib.pyplot as plt
import yfinance as yf  # requires yfinance - pip install yfinance
from requests.exceptions import ConnectionError, Timeout, TooManyRedirects
# Create your models here.


class User(models.Model):
    print('hello ehsan !')
    name = models.CharField(max_length=100, name='name')
    account = models.IntegerField(default=0, name='account')
    speed = models.IntegerField(default=1, name='speed')
    phone = models.CharField(default='09125459232', name='phone', max_length=11)
    last_mail_date = models.DateTimeField(name='last_mail_date')

    def __str__(self):
        return self.name

<<<<<<< HEAD
print('hi فاثقث')
=======
>>>>>>> 1

class Material(models.Model):
    name = models.CharField(name='name', max_length=100, default='')
    persian_name = models.CharField(name='persian_name', max_length=100, default='')
    price = models.FloatField(name="price")
    volume = models.FloatField(name='volume', default=0.0)
    update = models.DateTimeField(name='update')
    trading_fee = models.FloatField(name='trading_fee', default=0.01)
    state = models.BooleanField(name='state', default=True)
    hour_tendency = models.IntegerField(name='hour_tendency', default=0)
    hour_calc = models.DateTimeField(name='hour_calc', default=timezone.now)
    hour_conf = models.IntegerField(name='hour_conf', default=0)
    day_tendency = models.IntegerField(name='day_tendency', default=0)
    dat_calc = models.DateTimeField(name='day_calc', default=timezone.now)
    day_conf = models.IntegerField(name='day_conf', default=0)
    week_tendency = models.IntegerField(name='week_tendency', default=0)
    week_calc = models.DateTimeField(name='week_calc', default=timezone.now)
    week_conf = models.IntegerField(name='week_conf', default=0)

    def __str__(self):
        return self.name

    def make_ohl_cv(self,
                    start_time=timezone.now() - timezone.timedelta(days=30),
                    end_time=timezone.now(),
                    time_step='15Min'):
        """
        :param start_time: which time to start? django date time
        :param end_time: which time to end? django date time
        :param time_step: time steps choose from: ['15Min', '30S', '1D', ...]
        :return: open high low close volume DF
        """
        signals = self.signal_set.filter(date_time__gt=start_time, date_time__lt=end_time).order_by('date_time')
        date_times = pd.DatetimeIndex([signal.date_time for signal in signals])
        # print(date_times)
        price_series = pd.Series([signal.price for signal in signals], index=date_times)
        volume_series = pd.Series([signal.volume for signal in signals], index=date_times)
        df = pd.concat([price_series.resample(time_step).ohlc(), volume_series.resample(time_step).first()], axis=1)
        df.ffill(axis=0, inplace=True)
        df_new = df.rename(columns={0: 'Volume', 'open': 'Open', 'close': 'Close', 'high': 'High', 'low': 'Low'})
        df_new['DateTime'] = df.index
        df = df_new.set_index(pd.Index([i + 1 for i in range(len(df))]))
        df = df[['DateTime', 'Open', 'High', 'Low', 'Close', 'Volume']]
        df["DateTime"] = df["DateTime"].dt.strftime("%Y-%m-%d %H:%M:%S")
        # print(df)
        return df

    def give_support_resistance(self):
        df = self.make_ohl_cv(start_time=timezone.now()-timezone.timedelta(hours=8), time_step='1Min')
        print(df)
        # print(pd.Series(df['Close']))
        # this will serve as an example for security or index closing prices, or low and high prices
        # mins, maxs = trendln.calc_support_resistance(pd.Series(df['Close']))
        minimaIdxs, pmin, mintrend, minwindows = trendln.calc_support_resistance((pd.Series(df['Low']), None))
        print(minimaIdxs, pmin, mintrend, minwindows)
        minimaIdxs, pmin, mintrend, minwindows = trendln.calc_support_resistance((pd.Series(df['High']), None))
        print(minimaIdxs, pmin, mintrend, minwindows)
        print('hi')
        fig = trendln.plot_support_resistance((pd.Series(df['Close']), None))  #
        # requires matplotlib - pip install matplotlib
        plt.savefig('suppres.svg', format='svg')
        plt.show()


class Predictor(models.Model):
    material = models.ForeignKey(Material, on_delete=models.CASCADE)
    model_dir = models.CharField(name='model_dir', default='polls/trained/?.h5', max_length=100)
    time_frame = models.CharField(name='time_frame', default='1H', max_length=20)
    last_calc = models.DateTimeField(name='last_calc', default=timezone.now)
    input_size = models.IntegerField(name='input_size', default=24)
    type = models.CharField(name='type', default='RNN', max_length=20)
    unit = models.CharField(name='unit', default='dollar', max_length=20)

    def predict(self):
        """
        Inter your code
        :return:
        """
        df = ''
        if self.time_frame == '1H':
            size = self.input_size
            df = self.material.make_ohl_cv(start_time=timezone.now() - timezone.timedelta(hours=size), time_step='1H')
        elif self.time_frame == '1D':
            size = self.input_size
            df = self.material.make_ohl_cv(start_time=timezone.now() - timezone.timedelta(days=size), time_step='1D')


class Signal(models.Model):
    material = models.ForeignKey(Material, on_delete=models.CASCADE)
    price = models.FloatField(name='price')
    date_time = models.DateTimeField(name='date_time', default=timezone.now)
    volume = models.FloatField(name='volume', default=-1.0)
    """
    this elements can be added later
    'price_usd': '10226.7', 'price_btc': '1.0', '24h_volume_usd': '7585280000.0',
    'market_cap_usd': '172661078165', 'available_supply': '16883362.0', 
    'total_supply': '16883362.0', 'max_supply': '21000000.0', 
    'percent_change_1h': '0.67', 'percent_change_24h': '0.78', 
    'percent_change_7d': '-4.79'
    """

    def __str__(self):
        return self.material.name + '   ---->   ' + str(self.date_time) + '   --->   ' + str(self.price)


class Paired(models.Model):
    material1 = models.ForeignKey(Material, on_delete=models.CASCADE)
    name = models.CharField(name='name', max_length=100)
    price1 = models.FloatField(name='price1')
    price2 = models.FloatField(name='price2')
    date_time = models.DateTimeField(name='date_time')


class Wrapper(models.Model):
    material = models.ForeignKey(Material, name='material', on_delete=models.CASCADE)
    the_url = models.URLField(name='the_url', default='https://www.tgju.org')
    is_api = models.BooleanField(name='is_api', default=False)
    api_key = models.CharField(max_length=10, default='price_usd')
    kind = models.CharField(max_length=10, default='li')
    the_id = models.CharField(max_length=100, default='')

    def __str__(self):
        return self.the_url

    def get_information(self):
        if self.is_api:
            session = Session()
            try:
                response = session.get(self.the_url)
                data = response.json()
                ans = float(data['bpi']['USD']['rate_float'])
            except (ConnectionError, Timeout, TooManyRedirects) as e:
                print(e)
                return False

        else:

            response = django_req.get(url=str(self.the_url))
            page = response.content
            soup = BeautifulSoup(page, 'html.parser')
            my_object = soup.find(self.kind, id=self.the_id)
            ans = my_object.find('span').getText()
            ans = ans.replace(',', '')

        mat = self.material
        if float(mat.price) != float(ans):
            mat.price = float(ans)
            mat.update = timezone.now()
            mat.save()
            sig = Signal(material=self.material, price=float(ans), date_time=timezone.now())
            sig.save()


class Manager:
    def __init__(self, speed):
        self.speed = speed

    @staticmethod
    def give_difference(self, mat1, mat2):
        pairs = mat1.paired_set.filter(name=mat1.name + '---' + mat2.name)
        print('we have: ', len(pairs), ' data.')
        price1 = mat1.price
        price2 = mat2.price
        p1_on_p2 = [x.price1 / x.price2 for x in pairs]
        # hist = np.histogram(p1_on_p2, density=True)
        p1_on_p2_new = price1 / price2
        good_to_buy = 0
        good_to_sell = 0
        for old_data in p1_on_p2:
            if p1_on_p2_new < old_data:
                good_to_buy += 1
            else:
                good_to_sell += 1
        return good_to_buy / (good_to_buy + good_to_sell)


def give_distribution(mat1, mat2):
    signals_1 = mat1.signal_set.all()
    signals_2 = mat2.signal_set.all()
    pairs = []
    for signal_1 in signals_1:
        for signal_2 in signals_2:
            if (signal_1.date_time.date() == signal_2.date_time.date()) &\
                    (signal_1.date_time.hour == signal_2.date_time.hour) &\
                    (signal_1.date_time.minute == signal_2.date_time.minute):
                pairs.append(tuple([signal_1.price, signal_2.price]))
                paired = Paired(material1=mat1, name=mat1.name + '---' + mat2.name, price1=float(signal_1.price),
                                price2=float(signal_2.price), date_time=signal_1.date_time)
                paired.save()
    return pairs
