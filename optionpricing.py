import random
from collections import namedtuple

import numpy as np
from gym import spaces
from scipy.stats import norm


def compute_call(S, K, t, r, sigma):
    """
    基于BSM计算看涨期权的价格
    :param S: 标的资产价格
    :param K: 行权价
    :param t: 期限
    :param r: 无风险利率
    :param sigma: 波动率
    :return: 看涨期权的价格
    """
    if np.isclose(t, 0):
        return max(0, S - K)

    if t == 0:
        return max(0, S - K)

    d1 = (np.log(S / K) + (r + sigma ** 2 / 2) * t) / sigma / np.sqrt(t)
    d2 = d1 - sigma * np.sqrt(t)
    # norm.cdf：标准正态分布的累计概率分布函数
    call = S * norm.cdf(d1) - K * np.exp(-r * t) * norm.cdf(d2)

    return call


def compute_greeks(S, K, t, r, sigma):
    """
    计算希腊字母，风险
    :param S: 标的资产
    :param K: 行权价
    :param t: 期限
    :param r: 无风险利率
    :param sigma: 波动率
    :return: delta(期权价格关于标的资产价格的敏感度) gamma(Delta关于标的资产价格的微小变化的敏感性)
    """
    if np.isclose(t, 0):
        return 1, 0

    if t == 0:
        return 1, 0

    d1 = (np.log(S / K) + (r + sigma ** 2 / 2) * t) / sigma / np.sqrt(t)
    delta = norm.cdf(d1)
    gamma = (1 / np.sqrt(2 * np.pi)) * np.exp(-d1 ** 2 / 2) / (S * sigma * np.sqrt(t))

    return delta, gamma


def compute_pnl(init_portfolio, final_portfolio):
    """
    计算成本
    :param init_portfolio: 初始时刻的投资组合
    :param final_portfolio: 结束时刻的投资组合
    :return: 成本
    """
    # 初始时刻投资组合的价值
    init_wealth = compute_wealth(init_portfolio)
    # 结束时刻投资组合的价值
    final_wealth = compute_wealth(final_portfolio)

    # 成本
    return final_wealth - init_wealth


def compute_wealth(portfolio):
    """
    计算价值
    :param portfolio: 投资组合
    :return: 价值
    """
    # 期权价值
    # TODO
    option_value = portfolio.call * portfolio.m * portfolio.L
    # 标的资产价值 = 每一个标的资产的价值 * 数量
    stock_value = portfolio.S * portfolio.n
    # 现金
    cash = portfolio.cash

    return option_value + stock_value + cash


class OptionPricingEnv:
    def __init__(self, config):
        """
        期权定价环境
        :param config:
        config: Configuration dictionary with k:v as
            S: stock price (float)      标的资产价格
            T: days to maturity (int or list of ints)      到期时间
            L: number of option contracts (int)      期权合约数
            m: number of stocks per option (int)      每个期权的标的资产数
            n: number of stocks (int)      标的资产数
            K: strike price (float or list of floats)      行权价
            # TODO
            D: trading periods per day (int)
            mu: expected rate of return on the stock (float)      标的资产的预期回报
            sigma: volatility of stock (float)      标的资产的波动率
            r: risk free rate (float)      无风险利率
            # TODO
            ss: number of steps between trading periods (int)
            # 风险厌恶
            kappa: risk aversion (float)
        """
        self.config = config

        self.trading_days = 252
        # 24 hours
        self.day = 24 / self.trading_days
        self.lots = 1

        self.configured = False

    @property
    def call(self):
        # 看涨期权的价格
        return compute_call(self.S, self.K, self.t, self.r, self.sigma)

    @property
    def portfolio(self):
        # 投资组合
        # s: 标的资产
        # call: 看涨期权
        # n: 标的资产数
        # m: 每个期权的标的资产数
        # L: 期权合约数
        # cash: 现金
        return namedtuple('Portfolio', ['S', 'call', 'n', 'm', 'L', 'cash'])(self.S, self.call, self.n, self.m, self.L, self.cash)

    @property
    def stock_value(self):
        # 标的资产价值：每一个标的资产的价值 * 数量
        return self.n * self.S

    @property
    def option_value(self):
        # TODO
        return self.call * self.m * self.L

    @property
    def delta(self):
        # Delte(期权价格关于标的资产价格的敏感度)
        delta, gamma = compute_greeks(self.S, self.K, self.t, self.r, self.sigma)
        return delta

    def configure(self):
        self.S = self.config['S']
        try:
            self.T = random.choice(self.config['T'])
        except TypeError:
            self.T = self.config['T']

        self.L = self.config['L']
        self.m = self.config['m']
        self.n = self.config['n']
        try:
            self.K = random.choice(self.config['K'])
        except TypeError:
            self.K = self.config['K']

        # self.K = K
        self.D = self.config['D']
        self.mu = self.config['mu']
        self.sigma = self.config['sigma'] * np.sqrt(self.trading_days)  # Converting sigma/day to sigma/year
        self.r = self.config['r']
        self.ss = self.config['ss']
        self.kappa = self.config['kappa']
        self.multiplier = self.config['multiplier']
        self.ticksize = self.config['ticksize']

        self.S0 = self.S
        self.cash = 0

        # self.init_config = {k: v for k, v in locals().items() if k != 'self'}

        self.t = self.day * self.T
        self.steps = self.T * self.D
        self.dt = self.day / self.D / self.ss

        if not np.isclose(0, (self.t / self.dt) % 1):
            raise ValueError('Mismatch in "time to expiry" and "stochastic time step"')

        h = abs(self.L * self.m)
        l = -h
        num_actions = int((h - l) / self.lots + 1)

        self.high = h

        self.observation_space = spaces.Box(low=np.array([0, 0, 0, -np.inf]),
                                            high=np.array([np.inf, np.inf, np.inf, np.inf]))
        self.action_space = spaces.Discrete(num_actions)

        self.action_map = {i: int(l + i * self.lots) for i in range(self.action_space.n)}
        self.inv_action_map = {v: k for k, v in self.action_map.items()}

        self.configured = True
        self.done = False

    def step(self, action, stock_prices=None):
        """
        stock_prices: for deterministic evolution | dtype: list (even if single entry)
        """
        if not self.configured:
            raise NotImplementedError('Environment not configured')

        if self.done:
            return

        init_portfolio = self.portfolio

        num_stocks = self.action_map[action]
        self.n = self.n + num_stocks

        states = []
        calls = []
        deltas = []
        gammas = []

        self.cash = self.cash - self.S * num_stocks

        for i, period in enumerate(range(self.ss)):
            if stock_prices is not None:
                self.S = stock_prices[i]

            else:
                ds = self.mu * self.S * self.dt + self.sigma * self.S * np.random.normal() * np.sqrt(self.dt)
                self.S = self.S + ds

            self.t = max(0, self.t - self.dt)

            call = self.call
            delta, gamma = compute_greeks(self.S, self.K, self.t, self.r, self.sigma)

            calls.append(call)
            deltas.append(delta)
            gammas.append(gamma)

            states.append([self.S / self.S0, self.t, self.n / self.high, self.K / self.S0])

        self.steps -= 1

        cost = self.multiplier * self.ticksize * (abs(num_stocks) + 0.01 * num_stocks ** 2)
        self.cash -= cost

        pnl = compute_pnl(init_portfolio, self.portfolio)

        reward = (pnl - 0.5 * self.kappa * (pnl ** 2))

        info = {'pnl': pnl, 'dn': num_stocks, 'call': np.array(calls), 'delta': np.array(deltas),
                'gamma': np.array(gammas), 'cost': cost}

        self.done = self.steps == 0

        return np.array(states[-1], dtype=np.float32), reward, self.done, info

    def reset(self):
        self.configure()
        return np.array([self.S / self.S0, self.t, self.n / self.high, self.K / self.S0], dtype=np.float32)

    def render(self):
        pass

    def close(self):
        pass
