from django.db import models
import pandas as pd
from django.utils import timezone
from django.http import request as django_req
from bs4 import BeautifulSoup
from requests import Session
from requests.exceptions import ConnectionError, Timeout, TooManyRedirects
from tensorflow.keras.models import load_model
import joblib
import numpy as np
from binance.client import Client
from binance.enums import *
import ta
# Create your models here.


class User(models.Model):
    name = models.CharField(max_length=100, name='name')
    account = models.IntegerField(default=0, name='account')
    speed = models.IntegerField(default=1, name='speed')
    phone = models.CharField(default='09125459232', name='phone', max_length=11)
    last_mail_date = models.DateTimeField(name='last_mail_date')
    api_key = models.CharField(max_length=100, name='api_key',
                               default='api_key')
    secret_key = models.CharField(max_length=100, name='secret_key',
                                  default='secret_key')

    def __str__(self):
        return self.name


class Finance(models.Model):
    user = models.ForeignKey(User, on_delete=models.CASCADE)
    symbol = models.CharField(name='symbol', default='BTCUSDT', max_length=20)
    # todo: check the order responses to figure out the transactions are done perfectly

    def get_time(self):
        client = Client(self.user.api_key, self.user.secret_key)
        timestamp = client.get_server_time()
        return int(timestamp['serverTime']) / 1000

    def buy(self, percent=100):
        """

        :param percent: how much of your budget do you want to buy? 100 mean all of that
        :return:
        """
        client = Client(self.user.api_key, self.user.secret_key)

        order = client.order_market_buy(
            symbol=self.symbol,
            quantity=percent)
        return

    def sell(self, percent=100):
        """

        :param percent: how much of your asset do you want to sell? 100 mean all of that
        :return:
        """
        client = Client(self.user.api_key, self.user.secret_key)

        order = client.order_market_sell(
            symbol=self.symbol,
            quantity=percent)

        return

    def get_price(self):
        client = Client(self.user.api_key, self.user.secret_key)
        orders = client.get_all_tickers()
        for order in orders:
            if order['symbol'] == self.symbol:
                return float(order['price'])
        return False

    def buy_limit(self, limit, percent=100):
        """
        :param limit: in witch cost do you want to buy
        :param percent: how much of your budget do you want to buy? 100 mean all of that
        :return:
        """
        client = Client(self.user.api_key, self.user.secret_key)

        order = client.order_limit_buy(
            symbol=self.symbol,
            quantity=percent,
            price=str(limit))

    def sell_limit(self, limit, percent=100):
        """
        :param limit: in witch cost do you want to sell
        :param percent: how much of your asset do you want to sell? 100 mean all of that
        :return:
        """
        client = Client(self.user.api_key, self.user.secret_key)

        order = client.order_limit_sell(
            symbol=self.symbol,
            quantity=percent,
            price=str(limit))

    def buy_stop(self, stop, percent=100):
        """

        :param stop: the price of stop
        :param percent: how much of the budget? 100 means all of it
        :return:
        """
        client = Client(self.user.api_key, self.user.secret_key)

        order = client.create_oco_order(
            symbol=self.symbol,
            side=SIDE_BUY,
            stopLimitTimeInForce=TIME_IN_FORCE_GTC,
            quantity=percent,
            stopPrice=str(stop),
            price=str(stop))
        return

    def sell_stop(self, stop, percent=100):
        """

        :param stop: the price of stop
        :param percent: how much of the asset? 100 means all of it
        :return:
        """
        client = Client(self.user.api_key, self.user.secret_key)

        order = client.create_oco_order(
            symbol=self.symbol,
            side=SIDE_SELL,
            stopLimitTimeInForce=TIME_IN_FORCE_GTC,
            quantity=percent,
            stopPrice=str(stop),
            price=str(stop))
        return

    def cancel_stop(self):

        client = Client(self.user.api_key, self.user.secret_key)

        orders = client.get_open_orders(symbol=self.symbol)
        for order in orders:
            if order['symbol'] == self.symbol:
                the_answer = client.cancel_order(symbol=self.symbol, orderId=order['orderId'])
        return

    def give_ohlcv(self, interval='1m', size=12):
        """
        :param interval: it can be 1m for 1min and 1h for one hour if you want else, then you should define it.
        :param size: the size of the input size of predictor plus 1
        :return: a data_frame
        """
        client = Client(self.user.api_key, self.user.secret_key)
        if interval == '1h':
            my_interval = Client.KLINE_INTERVAL_1HOUR
            limit = (size + 1)
        elif interval == '1m':
            my_interval = Client.KLINE_INTERVAL_1MINUTE
            limit = (size + 1) * 60
        else:
            return False
        candles = client.get_klines(symbol=self.symbol, interval=my_interval, limit=limit)
        df = pd.DataFrame(candles, columns=["Open time", "Open", "High", "Low", "Close", "Volume", "Close time",
                                            "Quote asset volume", "Number of trades", "Taker buy base asset volume",
                                            "Taker buy quote asset volume", "Ignore"])
        return df


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
        df = df_new.set_index(pd.Index([i for i in range(len(df))]))
        df = df[['DateTime', 'Open', 'High', 'Low', 'Close', 'Volume']]
        df["DateTime"] = df["DateTime"].dt.strftime("%Y-%m-%d %H:%M:%S")
        return df


class Predictor(models.Model):
    material = models.ForeignKey(Material, on_delete=models.CASCADE)
    model_dir = models.CharField(name='model_dir', default='polls/trained/?.h5', max_length=100)
    i_scale = models.CharField(name='i_scale', default='polls/trained/I_scaler.gz', max_length=100)
    o_scale = models.CharField(name='o_scale', default='polls/trained/O_scaler.gz', max_length=100)
    time_frame = models.CharField(name='time_frame', default='1h', max_length=20)
    last_calc = models.DateTimeField(name='last_calc', default=timezone.now)
    input_size = models.IntegerField(name='input_size', default=24)
    type = models.CharField(name='type', default='RNN', max_length=20)
    unit = models.CharField(name='unit', default='dollar', max_length=20)
    upper = models.FloatField(name='upper', default=0)
    lower = models.FloatField(name='lower', default=0)

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if self.type == 'KNN':
            self.net = load_model(self.model_dir)
            self.i_scale = joblib.load(self.i_scale)
            self.o_scale = joblib.load(self.o_scale)
        elif self.type == 'DT':
            self.model = joblib.load(self.model_dir)


    @staticmethod
    def derivate(df):
        ind = df.index[0:-1]
        t = df.diff().iloc[1:]
        t.index = ind

        tt = df.iloc[0:-1]

        ttt = t.div(tt) * 100
        print('what the hell 4:', ttt)
        return ttt

    @staticmethod
    def make_inputs(df, test=True):
        """
        :param df: after doing ta.add_all... we give the result df to the input
        :param test: It should be True while predicting
        :return:
        """
        df.drop(['Open', 'Close', 'High', 'Low'], axis=1)
        inputs = []
        for index, data in df.iterrows():
            inputs.append(list(data))
        if test:
            return inputs
        else:
            return inputs[0:-1]

    def make_ohlcv(self, df):
        out = df[['Open', 'High', 'Low', 'Close', 'Volume']]
        out['Open'] = pd.to_numeric(out['Open'])
        out['High'] = pd.to_numeric(out['High'])
        out['Low'] = pd.to_numeric(out['Low'])
        out['Close'] = pd.to_numeric(out['Close'])
        out['Volume'] = pd.to_numeric(out['Volume'])
        out['Volume'] = out['Volume'] * out['Close']
        print(out.head())
        return out

    def predict(self, df=''):
        """
        Inter your code
        :param: df: the df should be appropriated for prediction in size and time framing
        :return:
        """
        if self.type == 'RNN':
            # todo: check that df exists?
            df = pd.to_numeric(df['Close']['mean'], downcast='float')
            arr = self.derivate(df).values.reshape(-1, 1)
            scaled_input = self.i_scale.transform(arr)
            the_input = np.reshape(scaled_input, (arr.shape[1], arr.shape[0], -1))
            prediction = self.net.predict(the_input)
            scaled_prediction = self.o_scale.inverse_transform(prediction)
            return scaled_prediction
        elif self.type == 'DT':
            df = self.make_ohlcv(df)
            df = ta.add_all_ta_features(df, open="Open", high="High", low="Low", close="Close",
                                        volume="Volume", fillna=True)
            the_input = self.make_inputs(df, test=True)
            predictions = self.model.predict(the_input)
            print('prediction is:', list(predictions[-1]))
            if list(predictions[-1]) in [[1, 0, 0, 0, 0, 0], [0, 1, 0, 0, 0, 0], [0, 0, 1, 0, 0, 0]]:
                return 1
            elif list(predictions[-1]) not in [[1, 0, 0, 0, 0, 0], [0, 1, 0, 0, 0, 0], [0, 0, 1, 0, 0, 0]]:
                return -1


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


class Trader(models.Model):
    predictor = models.ForeignKey(Predictor, on_delete=models.CASCADE)
    user = models.ForeignKey(User, on_delete=models.CASCADE, null=True, related_name='user')
    type = models.CharField(name='type', max_length=100)
    start_asset = models.FloatField(name='start_asset', default=0)  # it should be in dollar
    simulated_asset = models.FloatField(name='simulated_asset', default=0)
    real_mat_asset = models.FloatField(name='real_mat_asset', default=0)
    real_budget = models.FloatField(name='real_budget', default=0)
    start_date = models.DateTimeField(name='start_date', default=timezone.now)
    active = models.BooleanField(name='active', default=False)
    asset_chart_file = models.CharField(name='asset_chart_file', default='polls/chart/?.txt', max_length=100)

    def calculate(self):
        self.asset_chart_file = 'polls/chart/' + str(self.user.name) + '.txt'
        self.save()
        activities = self.activity_set.all()
        self.simulated_asset = activities[-1].price * activities[-1].mat_amount + activities[-1].budget
        with open(str(self.asset_chart_file), 'w') as file:
            file.flush()
            real_budget = self.start_asset
            real_mat_asset = 0
            simulated_asset = 0
            file.write(str(self.start_date) + ', ' + str(self.real_budget) + ', ' + str(True) + '\n')
            for activity in activities:
                file.write(str(activity.date_time) + ', ' + str(activity.price * activity.mat_amount + activity.budget)
                           + ', ' + str(activity.real) + '\n')
                simulated_asset = activity.price * activity.mat_amount + activity.budget
                if activity.real:
                    if activity.action == 'buy':
                        if real_budget != 0:
                            real_mat_asset = (real_budget / activity.price) * (1 - self.predictor.material.trading_fee)
                            real_budget = 0
                    elif activity.action == 'sell':
                        if real_mat_asset != 0:
                            real_budget = (real_mat_asset * activity.price) * (1 - self.predictor.material.trading_fee)
                            real_mat_asset = 0
                    file.write(str(activity.date_time) + ', ' +
                               str(activity.price * real_mat_asset + real_budget) +
                               ', ' + str(activity.real) + '\n')
            self.simulated_asset = simulated_asset
            self.real_mat_asset = real_mat_asset
            self.real_budget = real_budget

    def __str__(self):
        return str(self.predictor.material.name) + '---' + str(self.predictor.time_frame) + '---' + str(self.user.name)

    def trade(self, close, df=''):
        prediction = self.predictor.predict(df)
        print('prediction is:', prediction)
        if prediction > self.predictor.upper:
            self.buy(close)
            print('BUY')
        if prediction < self.predictor.lower:
            self.sell(close)
            print('SELL')

    def buy(self, close):
        speaker = self.user.finance_set.all()[0]
        if self.active:
            if self.type == '1':
                speaker.buy()
        price = speaker.get_price()
        if price:
            price = price
        else:
            price = close
        mat = self.predictor.material
        mat.price = price
        mat.save()
        self.real_mat_asset = self.real_mat_asset + ((self.real_budget / close) * (1 - mat.trading_fee))
        self.real_budget = 0
        self.save()
        record = Activity(trader=self, action='buy', date_time=timezone.now(), real=self.active, price=close,
                          budget=self.real_budget, mat_amount=self.real_mat_asset)
        record.save()

    def sell(self, close):
        speaker = self.user.finance_set.all()[0]
        if self.active:
            if self.type == '1':
                speaker.sell()
        price = speaker.get_price()
        if price:
            price = price
        else:
            price = close
        mat = self.predictor.material
        mat.price = price
        mat.save()
        self.real_mat_asset = self.real_mat_asset + ((self.real_budget / close) * (1 - mat.trading_fee))
        self.real_budget = 0
        self.save()
        record = Activity(trader=self, action='sell', date_time=timezone.now(), real=self.active, price=close,
                          budget=self.real_budget, mat_amount=self.real_mat_asset)
        record.save()


class Activity(models.Model):
    trader = models.ForeignKey(Trader, on_delete=models.CASCADE)
    action = models.CharField(name='action', max_length=3, default='buy')  # it should be 'buy' or 'sell'
    date_time = models.DateTimeField(name='date_time', default=timezone.now)
    real = models.BooleanField(name='real', default=False)
    price = models.FloatField(name='price', default=0)
    budget = models.FloatField(name='budget', default=0)
    mat_amount = models.FloatField(name='mat_amount', default=0)

    def __str__(self):
        return self.action + ' ' + str(self.trader.user.name) + str(self.date_time)


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
