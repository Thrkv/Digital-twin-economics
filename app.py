"""
Digital Twin Economics — app.py
================================
Монолитный файл с модульной ООП-архитектурой.

Запуск:
    streamlit run app.py

Зависимости:
    pip install streamlit scipy plotly pandas requests
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable

# Логгер для DataProvider — пишет в stderr / файл без примеси UI
_log = logging.getLogger("dte.data_provider")
if not _log.handlers:
    _handler = logging.StreamHandler()
    _handler.setFormatter(logging.Formatter(
        "%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    ))
    _log.addHandler(_handler)
    _log.setLevel(logging.INFO)

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import requests
import streamlit as st
from scipy.integrate import solve_ivp
from scipy.optimize import minimize, root_scalar


# ---------------------------------------------------------------------------
# Типы / перечисления
# ---------------------------------------------------------------------------

class Phase(Enum):
    """
    Фазы роста в модели Солоу с эндогенными технологиями.

    Attributes:
        EXTENSIVE:  Экстенсивный рост — накопление капитала, технологии стабильны.
        INTENSIVE:  Интенсивный рост — ТФП растёт, капитал эффективнее.
        RECESSION:  Рецессия — капитал амортизируется быстрее, ТФП падает.
    """
    EXTENSIVE = "extensive"
    INTENSIVE = "intensive"
    RECESSION = "recession"


class ShockType(Enum):
    """Типы шоков, поддерживаемых ScenarioManager."""
    DEMAND = "demand"
    SUPPLY = "supply"
    MONETARY = "monetary"
    FISCAL = "fiscal"


@dataclass
class EconomicState:
    """
    Снимок состояния экономики в момент времени t.

    Attributes:
        t:   Время (годы от начала симуляции).
        gdp: ВВП (млрд USD или индекс = 100).
        inflation: Инфляция (%).
        unemployment: Безработица (%).
        interest_rate: Ключевая ставка (%).
        phase: Текущая фаза цикла.
    """
    t: float = 0.0
    gdp: float = 100.0
    inflation: float = 2.0
    unemployment: float = 5.0
    interest_rate: float = 4.0
    phase: Phase = Phase.EXTENSIVE


@dataclass
class Shock:
    """
    Описание экономического шока.

    Attributes:
        shock_type: Тип шока.
        magnitude:  Величина (условные единицы).
        duration:   Длительность в годах.
        start_t:    Время начала (год симуляции).
        label:      Человекочитаемое название.
    """
    shock_type: ShockType
    magnitude: float
    duration: float
    start_t: float
    label: str = ""

    def __post_init__(self) -> None:
        if not self.label:
            self.label = f"{self.shock_type.value} shock @t={self.start_t}"


@dataclass
class Reform:
    """
    Структурная реформа — постоянное изменение одного параметра модели.

    Attributes:
        name:        Человекочитаемое название реформы.
        param:       Ключ в словаре params EconomicEngine (напр. "s", "r", "delta").
        delta:       Абсолютное изменение параметра (новое_значение = старое + delta).
        apply_at_t:  Год симуляции, начиная с которого реформа действует.
        description: Опциональное описание экономического смысла.
    """
    name: str
    param: str
    delta: float
    apply_at_t: float
    description: str = ""


@dataclass
class Modifier:
    """
    Внешний модификатор параметра — масштабирующий или аддитивный.

    Модификаторы применяются поверх базовых params и реформ,
    не затрагивая EconomicEngine напрямую.

    Attributes:
        name:   Идентификатор модификатора (уникальный в пределах ScenarioManager).
        param:  Ключ в словаре params (напр. "s", "r", "A_critical").
        mode:   Режим применения:
                - "add"   → param += value
                - "scale" → param *= value
        value:  Числовое значение изменения.
        label:  Отображаемое название (для UI и логов).
    """
    name: str
    param: str
    mode: str          # "add" | "scale"
    value: float
    label: str = ""

    def __post_init__(self) -> None:
        if self.mode not in ("add", "scale"):
            raise ValueError(f"Modifier.mode должен быть 'add' или 'scale', получено: {self.mode!r}")
        if not self.label:
            self.label = self.name


# ---------------------------------------------------------------------------
# Предопределённые шоки
# ---------------------------------------------------------------------------

@dataclass
class ShockTemplate:
    """
    Шаблон шока — полное описание одного типа кризиса.

    Шаблоны хранятся в ``SHOCK_CATALOG`` и не содержат пользовательских
    интервалов (start_t, duration). Те передаются отдельно при создании Shock.

    Attributes:
        key:            Уникальный идентификатор (напр. "capital_crisis").
        label:          Человекочитаемое название.
        description:    Экономический смысл шока.
        param_overrides: Абсолютные изменения параметров в период шока.
                         Ключи — те же, что в DEFAULT_PARAMS.
        phase_forced:   Если задан — фаза принудительно устанавливается
                        в этот период (None = определяется автоматически).
    """
    key: str
    label: str
    description: str
    param_overrides: dict[str, float]
    phase_forced: Phase | None = None


# Каталог доступных шоков — единственное место с «магическими» числами шоков
SHOCK_CATALOG: dict[str, ShockTemplate] = {
    "no_shock": ShockTemplate(
        key="no_shock",
        label="Без шока",
        description="Базовый сценарий без внешних нарушений.",
        param_overrides={},
        phase_forced=None,
    ),
    "capital_crisis": ShockTemplate(
        key="capital_crisis",
        label="Капитальный кризис",
        description=(
            "Резкое сокращение инвестиций и ускоренная амортизация. "
            "Моделирует финансовый кризис или коллапс кредитного рынка."
        ),
        param_overrides={
            "s_recession_factor":     0.25,   # сбережения падают до 25% от нормы
            "delta_recession_factor": 3.50,   # амортизация в 3.5× быстрее
            "A_decay_recession":      0.03,   # TFP снижается умеренно
        },
        phase_forced=Phase.RECESSION,
    ),
    "tech_stagnation": ShockTemplate(
        key="tech_stagnation",
        label="Технологическая стагнация",
        description=(
            "Прекращение роста TFP и снижение эффективности R&D. "
            "Моделирует исчерпание технологической парадигмы."
        ),
        param_overrides={
            "r":                  0.0,    # рост TFP останавливается
            "A_decay_recession":  0.08,   # TFP активно деградирует
            "s_recession_factor": 0.60,   # сбережения сокращаются умеренно
            "delta_recession_factor": 1.8,
        },
        phase_forced=Phase.RECESSION,
    ),
}


# ---------------------------------------------------------------------------
# 1. EconomicEngine
# ---------------------------------------------------------------------------

# Параметры по умолчанию — полный словарь
DEFAULT_PARAMS: dict[str, float] = {
    # --- Модель Солоу ---
    "alpha":      0.33,   # эластичность выпуска по капиталу (доля капитала в ВВП)
    "delta":      0.05,   # норма амортизации капитала (5% в год)
    "n":          0.01,   # темп роста населения (1% в год)
    "s":          0.20,   # норма сбережений (20% ВВП)
    # --- Технологии ---
    "gamma":      0.50,   # эластичность выпуска по TFP в интенсивной фазе
    "r":          0.02,   # темп роста TFP в интенсивной фазе (2% в год)
    "A_critical": 1.50,   # порог TFP для перехода extensive → intensive
    # --- Рецессия ---
    "s_recession_factor":     0.40,   # норма сбережений в рецессию = s * 0.4
    "delta_recession_factor": 2.50,   # амортизация в рецессию = delta * 2.5
    "A_decay_recession":      0.05,   # темп падения TFP в рецессию (5% в год)
    # --- Численные ограничения ---
    "k_min": 1e-6,   # нижний порог капиталовооружённости (защита от 0)
    "A_min": 1e-6,   # нижний порог TFP
}


class EconomicEngine:
    """
    Математическое ядро Digital Twin Economics.

    Реализует расширенную модель Солоу с эндогенными технологиями
    и тремя фазами роста: extensive, intensive, recession.

    Система состояния:
        y = [k, A]
        k — капиталовооружённость (капитал на одного работника)
        A — уровень технологий (TFP, Total Factor Productivity)

    Уравнения по фазам
    ------------------
    Extensive:
        dk/dt = s * A * k^alpha - (delta + n) * k
        dA/dt = 0

    Intensive:
        dk/dt = s * A^gamma * k^alpha - (delta + n) * k
        dA/dt = r * A

    Recession:
        dk/dt = (s * s_recession_factor) * k^alpha - (delta * delta_recession_factor + n) * k
        dA/dt = -A_decay_recession * A

    Переключение фаз (state machine):
        - RECESSION  если t попадает в интервал хотя бы одного активного Shock
        - INTENSIVE  если A >= A_critical  (и не рецессия)
        - EXTENSIVE  иначе

    Usage::

        engine = EconomicEngine(t_span=(0, 30), dt=0.25)
        params = {**DEFAULT_PARAMS, "s": 0.25, "r": 0.03}
        df = engine.solve(k0=2.0, A0=1.0, params=params, shocks=shocks)
        gdp_df = engine.compute_gdp(df, params)
        state  = engine.get_state(t=10.0, df=df, params=params, shocks=shocks)
    """

    def __init__(
        self,
        t_span: tuple[float, float] = (0.0, 20.0),
        dt: float = 0.25,
    ) -> None:
        """
        Args:
            t_span: (t_start, t_end) в годах.
            dt:     Шаг вывода временного ряда (не шаг интегратора — тот адаптивный).
        """
        self.t_span = t_span
        self.dt = dt
        # Последний успешный результат интегрирования (кэш для get_state)
        self._last_sol: Any = None

    # ==================================================================
    # Публичный интерфейс
    # ==================================================================

    def solve(
        self,
        k0: float,
        A0: float,
        params: dict[str, float] | None = None,
        shocks: list[Shock] | None = None,
        dynamic_params: Callable[[float, dict[str, float]], dict[str, float]] | None = None,
    ) -> pd.DataFrame:
        """
        Интегрирует систему ОДУ и возвращает временной ряд.

        Args:
            k0:     Начальная капиталовооружённость (k > 0).
            A0:     Начальный уровень технологий (A > 0).
            params: Словарь параметров модели (см. DEFAULT_PARAMS).
                    Отсутствующие ключи берутся из DEFAULT_PARAMS.
            shocks: Список шоков — определяют интервалы рецессии.
            dynamic_params: Необязательная функция ``f(t, params) -> overrides``,
                    вызываемая на каждом шаге интегратора для получения
                    time-varying переопределений параметров (используется,
                    например, ``AdvancedShockManager`` для гладких/
                    циклических шоков). Если ``None`` (по умолчанию) —
                    поведение полностью идентично прежнему: параметры
                    статичны на протяжении всей траектории.

        Returns:
            DataFrame с колонками:
                t        — время (годы)
                k        — капиталовооружённость
                A        — уровень технологий
                phase    — строковое название фазы ('extensive' / 'intensive' / 'recession')
                dk_dt    — производная dk/dt в данной точке
                dA_dt    — производная dA/dt в данной точке

        Raises:
            ValueError: если k0 <= 0 или A0 <= 0.
        """
        p = {**DEFAULT_PARAMS, **(params or {})}
        shocks = shocks or []

        if k0 <= 0:
            raise ValueError(f"k0 должна быть > 0, получено {k0}")
        if A0 <= 0:
            raise ValueError(f"A0 должна быть > 0, получено {A0}")

        y0 = [float(k0), float(A0)]
        t_eval = self._make_t_eval()

        def rhs(t: float, y: list[float]) -> list[float]:
            return self._rhs(t, y, p, shocks, dynamic_params)

        try:
            sol = solve_ivp(
                rhs,
                self.t_span,
                y0,
                t_eval=t_eval,
                method="RK45",
                rtol=1e-6,
                atol=1e-9,
                dense_output=False,
            )
        except Exception as exc:
            raise RuntimeError(f"solve_ivp завершился с ошибкой: {exc}") from exc

        if not sol.success:
            raise RuntimeError(f"solve_ivp не сошёлся: {sol.message}")

        self._last_sol = sol
        return self._build_dataframe(sol, p, shocks, dynamic_params)

    def compute_gdp(
        self,
        df: pd.DataFrame,
        params: dict[str, float] | None = None,
        L0: float = 1.0,
        n: float | None = None,
    ) -> pd.DataFrame:
        """
        Вычисляет ВВП из траектории (k, A).

        Производственная функция Кобба–Дугласа:
            Y(t) = A(t) * K(t)^alpha * L(t)^(1-alpha)
            K(t) = k(t) * L(t)
            L(t) = L0 * exp(n * t)

        Args:
            df:     DataFrame из solve() с колонками t, k, A.
            params: Словарь параметров (используются alpha, n).
            L0:     Начальный размер рабочей силы (нормировка).
            n:      Темп роста населения (переопределяет params["n"]).

        Returns:
            df с добавленными колонками: L, K, Y, y (ВВП на работника).
        """
        p = {**DEFAULT_PARAMS, **(params or {})}
        alpha = p["alpha"]
        pop_growth = n if n is not None else p["n"]

        out = df.copy()
        out["L"] = L0 * (out["t"] * pop_growth).map(lambda x: __import__("math").exp(x))
        out["K"] = out["k"] * out["L"]
        out["Y"] = out["A"] * out["K"] ** alpha * out["L"] ** (1.0 - alpha)
        out["y"] = out["Y"] / out["L"]   # ВВП на работника
        return out

    def get_state(
        self,
        t: float,
        df: pd.DataFrame,
        params: dict[str, float] | None = None,
        shocks: list[Shock] | None = None,
    ) -> dict[str, Any]:
        """
        Возвращает полный снимок состояния экономики в момент t.

        Args:
            t:      Момент времени (год).
            df:     DataFrame из solve().
            params: Параметры модели.
            shocks: Список шоков.

        Returns:
            Словарь:
                t, k, A, phase, dk_dt, dA_dt, gdp_per_worker
        """
        p = {**DEFAULT_PARAMS, **(params or {})}
        shocks = shocks or []

        # Ближайшая точка временного ряда
        idx = (df["t"] - t).abs().idxmin()
        row = df.iloc[idx]

        k = float(row["k"])
        A = float(row["A"])
        phase = self._determine_phase(float(row["t"]), k, A, p, shocks)
        dk, dA = self._phase_equations(float(row["t"]), k, A, phase, p)

        # ВВП на работника в данной точке (L нормирован к 1)
        gdp_per_worker = A * (k ** p["alpha"])

        return {
            "t":             float(row["t"]),
            "k":             k,
            "A":             A,
            "phase":         phase.value,
            "dk_dt":         dk,
            "dA_dt":         dA,
            "gdp_per_worker": gdp_per_worker,
        }

    # ==================================================================
    # State machine
    # ==================================================================

    def _determine_phase(
        self,
        t: float,
        k: float,
        A: float,
        p: dict[str, float],
        shocks: list[Shock],
    ) -> Phase:
        """
        Определяет фазу роста по правилам приоритета:
            1. RECESSION — если t находится внутри интервала хотя бы одного шока.
            2. INTENSIVE — если A >= A_critical.
            3. EXTENSIVE — во всех остальных случаях.

        Args:
            t:      Текущее время.
            k:      Текущая капиталовооружённость.
            A:      Текущий уровень технологий.
            p:      Параметры модели.
            shocks: Список шоков.

        Returns:
            Phase enum.
        """
        # Приоритет 1: рецессия (активный шок)
        for shock in shocks:
            if shock.start_t <= t < shock.start_t + shock.duration:
                return Phase.RECESSION

        # Приоритет 2: интенсивный рост
        if A >= p["A_critical"]:
            return Phase.INTENSIVE

        # Приоритет 3: экстенсивный рост
        return Phase.EXTENSIVE

    # ==================================================================
    # Система ОДУ
    # ==================================================================

    def _rhs(
        self,
        t: float,
        y: list[float],
        p: dict[str, float],
        shocks: list[Shock],
        dynamic_params: Callable[[float, dict[str, float]], dict[str, float]] | None = None,
    ) -> list[float]:
        """
        Правая часть системы ОДУ — вызывается solve_ivp на каждом шаге.

        Защиты:
        - k и A зажимаются снизу на k_min / A_min (предотвращает NaN при k→0).
        - Результат проверяется на NaN; при обнаружении возвращает нули.

        Args:
            t:      Текущее время.
            y:      Вектор состояния [k, A].
            p:      Параметры модели.
            shocks: Список шоков.
            dynamic_params: Опциональная f(t, p) -> overrides для
                    time-varying параметров (см. ``EconomicEngine.solve``).

        Returns:
            [dk/dt, dA/dt]
        """
        import math

        k = max(float(y[0]), p["k_min"])
        A = max(float(y[1]), p["A_min"])

        p_t = p if dynamic_params is None else {**p, **dynamic_params(t, p)}

        phase = self._determine_phase(t, k, A, p_t, shocks)
        dk, dA = self._phase_equations(t, k, A, phase, p_t)

        # Защита от NaN / Inf
        if not math.isfinite(dk):
            dk = 0.0
        if not math.isfinite(dA):
            dA = 0.0

        return [dk, dA]

    def _phase_equations(
        self,
        t: float,
        k: float,
        A: float,
        phase: Phase,
        p: dict[str, float],
    ) -> tuple[float, float]:
        """
        Вычисляет производные (dk/dt, dA/dt) для заданной фазы.

        Extensive:
            dk/dt = s * A * k^alpha - (delta + n) * k
            dA/dt = 0

        Intensive:
            dk/dt = s * A^gamma * k^alpha - (delta + n) * k
            dA/dt = r * A

        Recession:
            dk/dt = (s * s_recession_factor) * k^alpha
                    - (delta * delta_recession_factor + n) * k
            dA/dt = -A_decay_recession * A

        Args:
            t:     Текущее время (не используется явно, зарезервировано).
            k:     Капиталовооружённость (уже зажата снизу).
            A:     Уровень технологий (уже зажат снизу).
            phase: Активная фаза.
            p:     Параметры модели.

        Returns:
            (dk_dt, dA_dt)
        """
        alpha = p["alpha"]
        delta = p["delta"]
        n     = p["n"]
        s     = p["s"]

        k_alpha = k ** alpha  # k^alpha — общий множитель

        if phase == Phase.EXTENSIVE:
            dk = s * A * k_alpha - (delta + n) * k
            dA = 0.0

        elif phase == Phase.INTENSIVE:
            gamma = p["gamma"]
            r     = p["r"]
            dk = s * (A ** gamma) * k_alpha - (delta + n) * k
            dA = r * A

        elif phase == Phase.RECESSION:
            s_rec   = s * p["s_recession_factor"]
            dep_rec = delta * p["delta_recession_factor"] + n
            dk = s_rec * k_alpha - dep_rec * k
            dA = -p["A_decay_recession"] * A

        else:
            # Неизвестная фаза — нулевая динамика (fallback)
            dk, dA = 0.0, 0.0

        return dk, dA

    # ==================================================================
    # Вспомогательные методы
    # ==================================================================

    def _make_t_eval(self) -> list[float]:
        """Формирует равномерную сетку точек вывода."""
        t0, t1 = self.t_span
        n = int(round((t1 - t0) / self.dt)) + 1
        return [t0 + i * self.dt for i in range(n)]

    def _build_dataframe(
        self,
        sol: Any,
        p: dict[str, float],
        shocks: list[Shock],
        dynamic_params: Callable[[float, dict[str, float]], dict[str, float]] | None = None,
    ) -> pd.DataFrame:
        """
        Конвертирует OdeResult в DataFrame, добавляет phase и производные.

        Также применяет постпроцессинг:
        - зажимает k и A снизу (k_min, A_min);
        - заменяет NaN нулями.

        Args:
            sol:    Результат solve_ivp.
            p:      Параметры модели.
            shocks: Список шоков.
            dynamic_params: Опциональная f(t, p) -> overrides — та же
                    функция, что была передана в ``solve()`` (см. её docstring).

        Returns:
            DataFrame с колонками: t, k, A, phase, dk_dt, dA_dt.
        """
        import numpy as np

        k_arr = np.clip(sol.y[0], p["k_min"], None)
        A_arr = np.clip(sol.y[1], p["A_min"], None)

        # Заменяем NaN нулями
        k_arr = np.where(np.isfinite(k_arr), k_arr, p["k_min"])
        A_arr = np.where(np.isfinite(A_arr), A_arr, p["A_min"])

        phases, dk_list, dA_list = [], [], []
        for t_i, k_i, A_i in zip(sol.t, k_arr, A_arr):
            p_i = p if dynamic_params is None else {**p, **dynamic_params(t_i, p)}
            ph = self._determine_phase(t_i, k_i, A_i, p_i, shocks)
            dk, dA = self._phase_equations(t_i, k_i, A_i, ph, p_i)
            phases.append(ph.value)
            dk_list.append(dk)
            dA_list.append(dA)

        return pd.DataFrame({
            "t":     sol.t,
            "k":     k_arr,
            "A":     A_arr,
            "phase": phases,
            "dk_dt": dk_list,
            "dA_dt": dA_list,
        })


# ---------------------------------------------------------------------------
# 1.5 StochasticEngine — Монте-Карло / SDE (Euler–Maruyama)
# ---------------------------------------------------------------------------

@dataclass
class MonteCarloResult:
    """
    Результат симуляции Монте-Карло от ``StochasticEngine.run_monte_carlo()``.

    Attributes:
        t:          Временная сетка (годы), shape (T,).
        k_paths:    Траектории капиталовооружённости, shape (n_runs, T).
        A_paths:    Траектории TFP, shape (n_runs, T) (шум не добавляется —
                    только детерминированный дрейф по фазам).
        y_paths:    Траектории ВВП на работника y = A * k^alpha, shape (n_runs, T).
        quantiles:  {"k": {"p2_5", "p50", "p97_5"}, "y": {...}} — перцентили
                    по траекториям в каждый момент времени (np.ndarray, shape (T,)).
        sigma:      Волатильность σ, использованная в симуляции.
        n_runs:     Число прогонов N.
        dt:         Шаг интегрирования Эйлера–Маруямы (годы).
        seed:       Seed генератора случайных чисел (для воспроизводимости).
    """
    t: "np.ndarray"
    k_paths: "np.ndarray"
    A_paths: "np.ndarray"
    y_paths: "np.ndarray"
    quantiles: dict[str, dict[str, "np.ndarray"]]
    sigma: float
    n_runs: int
    dt: float
    seed: int | None = None


class StochasticEngine:
    """
    Стохастическое расширение модели: симуляция Монте-Карло методом
    Эйлера–Маруямы (Euler–Maruyama) с мультипликативным винеровским шумом
    в динамике капитала (геометрическое броуновское движение поверх
    детерминированного дрейфа Солоу).

    ВАЖНО: этот класс НЕ изменяет и не вызывает ``EconomicEngine.solve()``
    (то есть основной детерминированный солвер ``scipy.integrate.solve_ivp``
    остаётся нетронутым). ``StochasticEngine`` реализует полностью
    независимый численный метод — явную схему Эйлера–Маруямы с
    фиксированным шагом — переиспользуя только *описание* фазовой
    динамики (``EconomicEngine._phase_equations`` / ``_determine_phase``)
    как детерминированный дрейф SDE.

    Математика (метод Эйлера–Маруямы)
    ----------------------------------
    Для каждой траектории i = 1..N::

        k_i(t+Δt) = k_i(t) + drift(k_i(t), A_i(t), t) · Δt
                            + σ · k_i(t) · sqrt(Δt) · ξ_i(t),   ξ_i(t) ~ N(0,1)

        A_i(t+Δt) = A_i(t) + dA/dt(k_i(t), A_i(t), t) · Δt   (без шума)

    где drift(k, A, t) = dk/dt согласно активной фазе роста
    (extensive / intensive / recession) — та же экономическая логика,
    что и в детерминированной модели. Шум добавлен только к капиталу k:
    это отражает волатильность инвестиций/нормы сбережений, тогда как
    уровень технологий A считается более инерционной величиной.

    Usage::

        engine = EconomicEngine(t_span=(0, 30), dt=0.5)
        se = StochasticEngine(engine)
        mc = se.run_monte_carlo(
            k0=2.0, A0=1.0, params=params, shocks=shocks,
            sigma=0.05, n_runs=100, seed=42,
        )
        # mc.t, mc.k_paths, mc.y_paths, mc.quantiles — готовы для PlotBuilder
    """

    def __init__(self, engine: EconomicEngine) -> None:
        """
        Args:
            engine: EconomicEngine, у которого переиспользуются ``t_span``
                    и ``dt`` (шаг Эйлера–Маруямы = ``engine.dt``), а также
                    фазовая динамика (``_phase_equations``/``_determine_phase``).
                    Сам объект не мутируется и его ``solve()`` не вызывается.
        """
        self.engine = engine

    def run_monte_carlo(
        self,
        k0: float,
        A0: float,
        params: dict[str, float] | None = None,
        shocks: list[Shock] | None = None,
        sigma: float = 0.05,
        n_runs: int = 100,
        seed: int | None = None,
    ) -> MonteCarloResult:
        """
        Запускает N траекторий методом Эйлера–Маруямы.

        Args:
            k0, A0:  Начальные условия (общие для всех траекторий).
            params:  Параметры модели (см. DEFAULT_PARAMS); отсутствующие
                     ключи берутся из DEFAULT_PARAMS.
            shocks:  Список шоков — определяют интервалы рецессии
                     (детерминированно, одинаково для всех траекторий).
            sigma:   Волатильность σ ≥ 0 (0 → все траектории совпадают
                     с детерминированной).
            n_runs:  Число траекторий N ≥ 1.
            seed:    Seed для ``numpy.random.default_rng`` (воспроизводимость).

        Returns:
            MonteCarloResult с полными траекториями и квантилями.

        Raises:
            ValueError: при некорректных k0, A0, sigma или n_runs.
        """
        p = {**DEFAULT_PARAMS, **(params or {})}
        shocks = shocks or []

        if k0 <= 0:
            raise ValueError(f"k0 должна быть > 0, получено {k0}")
        if A0 <= 0:
            raise ValueError(f"A0 должна быть > 0, получено {A0}")
        if sigma < 0:
            raise ValueError(f"sigma должна быть >= 0, получено {sigma}")
        if n_runs < 1:
            raise ValueError(f"n_runs должно быть >= 1, получено {n_runs}")

        t0, t1 = self.engine.t_span
        dt = float(self.engine.dt)
        n_steps = max(int(round((t1 - t0) / dt)), 1)
        t_grid = np.array([t0 + i * dt for i in range(n_steps + 1)], dtype=float)

        rng = np.random.default_rng(seed)
        sqrt_dt = math.sqrt(dt)

        k = np.full(n_runs, float(k0), dtype=float)
        A = np.full(n_runs, float(A0), dtype=float)

        k_paths = np.empty((n_runs, n_steps + 1), dtype=float)
        A_paths = np.empty((n_runs, n_steps + 1), dtype=float)
        k_paths[:, 0] = k
        A_paths[:, 0] = A

        alpha = p["alpha"]
        delta = p["delta"]
        n_pop = p["n"]
        s     = p["s"]
        gamma = p["gamma"]
        r     = p["r"]
        s_rec   = s * p["s_recession_factor"]
        dep_rec = delta * p["delta_recession_factor"] + n_pop
        k_min, A_min = p["k_min"], p["A_min"]

        for i in range(1, n_steps + 1):
            t = t_grid[i - 1]

            k_c = np.clip(k, k_min, None)
            A_c = np.clip(A, A_min, None)

            # Фаза: рецессия — общая для всех траекторий (зависит только от t);
            # extensive/intensive — индивидуальна для каждой траектории (зависит от A).
            in_shock = any(
                sh.start_t <= t < sh.start_t + sh.duration for sh in shocks
            )
            mask_rec = np.full(n_runs, in_shock, dtype=bool)
            mask_int = (~mask_rec) & (A_c >= p["A_critical"])
            mask_ext = (~mask_rec) & (~mask_int)

            dk = np.empty(n_runs, dtype=float)
            dA = np.empty(n_runs, dtype=float)

            k_alpha_c = k_c ** alpha

            # Extensive: dk = s*A*k^alpha - (delta+n)*k ; dA = 0
            m = mask_ext
            dk[m] = s * A_c[m] * k_alpha_c[m] - (delta + n_pop) * k_c[m]
            dA[m] = 0.0

            # Intensive: dk = s*A^gamma*k^alpha - (delta+n)*k ; dA = r*A
            m = mask_int
            dk[m] = s * (A_c[m] ** gamma) * k_alpha_c[m] - (delta + n_pop) * k_c[m]
            dA[m] = r * A_c[m]

            # Recession: dk = s_rec*k^alpha - dep_rec*k ; dA = -A_decay_recession*A
            m = mask_rec
            dk[m] = s_rec * k_alpha_c[m] - dep_rec * k_c[m]
            dA[m] = -p["A_decay_recession"] * A_c[m]

            # --- Euler–Maruyama: детерминированный дрейф + винеровский шум по k ---
            xi = rng.standard_normal(n_runs)
            k_new = k_c + dk * dt + sigma * k_c * sqrt_dt * xi
            A_new = A_c + dA * dt

            # Защита от NaN / Inf / ухода в отрицательную область
            k_new = np.clip(k_new, k_min, None)
            A_new = np.clip(A_new, A_min, None)
            k_new = np.where(np.isfinite(k_new), k_new, k_min)
            A_new = np.where(np.isfinite(A_new), A_new, A_min)

            k, A = k_new, A_new
            k_paths[:, i] = k
            A_paths[:, i] = A

        y_paths = A_paths * np.power(k_paths, alpha)

        quantiles = {
            "k": self._quantile_bands(k_paths),
            "y": self._quantile_bands(y_paths),
        }

        return MonteCarloResult(
            t=t_grid,
            k_paths=k_paths,
            A_paths=A_paths,
            y_paths=y_paths,
            quantiles=quantiles,
            sigma=float(sigma),
            n_runs=int(n_runs),
            dt=dt,
            seed=seed,
        )

    @staticmethod
    def _quantile_bands(paths: "np.ndarray") -> dict[str, "np.ndarray"]:
        """
        Считает 2.5%, 50% и 97.5% перцентили по траекториям в каждый момент t
        (т.е. вдоль axis=0 — по прогонам), формируя 95% доверительный интервал.

        Args:
            paths: Матрица траекторий, shape (n_runs, T).

        Returns:
            {"p2_5": np.ndarray(T,), "p50": np.ndarray(T,), "p97_5": np.ndarray(T,)}
        """
        return {
            "p2_5":  np.percentile(paths, 2.5, axis=0),
            "p50":   np.percentile(paths, 50.0, axis=0),
            "p97_5": np.percentile(paths, 97.5, axis=0),
        }


# ---------------------------------------------------------------------------
# 1.7 QualitativeAnalyzer — стационарные точки, матрица Якоби, устойчивость
# ---------------------------------------------------------------------------

@dataclass
class StabilityResult:
    """
    Результат анализа устойчивости от ``QualitativeAnalyzer.analyze()``.

    Attributes:
        k_star:            Стационарная точка k* (замкнутая формула).
        k_star_root_scalar: k*, найденное независимо через
                            ``scipy.optimize.root_scalar`` (кросс-проверка).
                            None, если численный поиск не сошёлся.
        root_scalar_converged: Флаг сходимости численной кросс-проверки.
        A_star:            Значение A, при котором зафиксирована точка (A
                            рассматривается как «замороженный» параметр —
                            квазистатический анализ, т.к. в интенсивной фазе
                            dA/dt = r·A ≠ 0 и глобальной стационарной точки
                            по A не существует).
        phase:             Фаза роста, для которой построена линеаризация
                            ('extensive' | 'intensive' | 'recession').
        params_snapshot:   Копия параметров модели, использованных в анализе.
        jacobian:          Матрица Якоби J (2×2, numpy.ndarray).
        eigenvalues:       Собственные значения [λ1, λ2] (могут быть комплексными).
        trace:             След J (λ1 + λ2).
        determinant:       Определитель J (λ1 · λ2).
        stability_type:    Короткий вердикт ("Устойчивый узел / Аттрактор",
                            "Неустойчивый источник", "Седловая точка", ...).
        verdict:           Развёрнутое текстовое объяснение для UI.
    """
    k_star: float
    k_star_root_scalar: float | None
    root_scalar_converged: bool
    A_star: float
    phase: str
    params_snapshot: dict[str, float]
    jacobian: "np.ndarray"
    eigenvalues: list[complex]
    trace: float
    determinant: float
    stability_type: str
    verdict: str


class QualitativeAnalyzer:
    """
    Независимый аналитический блок: находит стационарные точки k*,
    строит матрицу Якоби линеаризованной системы в (k*, A*) и определяет
    тип устойчивости по Ляпунову через знаки вещественных частей
    собственных значений.

    Не запускает интегрирование (``EconomicEngine.solve`` /
    ``scipy.integrate.solve_ivp``) и не требует прогона симуляции — работает
    чисто аналитически с параметрами модели p и «замороженным» уровнем A*.

    Квазистатическая интерпретация
    -------------------------------
    В интенсивной фазе технологии растут неограниченно (dA/dt = r·A), то
    есть глобальной стационарной точки по A не существует. Поэтому анализ
    выполняется в духе разделения временных масштабов (timescale
    separation): для «замороженного» на момент анализа значения A* ищется
    точка k*(A*), в которой dk/dt = 0, и строится линеаризация системы
    вокруг (k*, A*). Это стандартный приём качественного анализа Solow-типа
    моделей с экзогенным/полуэндогенным ростом ТФП — k быстро подстраивается
    к «движущейся мишени» k*(A(t)), тогда как A эволюционирует медленнее.

    Уравнения по фазам
    -------------------
    Extensive   (dA/dt = 0):
        dk/dt = s·A·k^α - (δ+n)·k
        k*    = (s·A / (δ+n))^(1/(1-α))

    Intensive   (dA/dt = r·A):
        dk/dt = s·A^γ·k^α - (δ+n)·k
        k*    = (s·A^γ / (δ+n))^(1/(1-α))

    Recession   (dA/dt = -A_decay_recession·A):
        dk/dt = s_rec·k^α - dep_rec·k      (не зависит от A)
        k*    = (s_rec / dep_rec)^(1/(1-α))

    Матрица Якоби (интенсивная фаза, как в ТЗ)::

        J = [[ α·s·A^γ·k*^(α-1) - (δ+n),   γ·s·A^(γ-1)·k*^α ],
             [ 0,                           r                 ]]

    Классификация по собственным значениям λ1, λ2:
        - Re(λ1) < 0 и Re(λ2) < 0  → устойчивый узел / фокус (аттрактор)
        - Re(λ1) > 0 и Re(λ2) > 0  → неустойчивый источник
        - разные знаки Re(λ)       → седловая точка
        - Re(λ) ≈ 0                → вырожденный / нейтральный случай

    Usage::

        analyzer = QualitativeAnalyzer(params=params_scen)
        result = analyzer.analyze(A_star=1.2, phase="intensive")
        print(result.k_star, result.eigenvalues, result.stability_type)
    """

    _VALID_PHASES = ("extensive", "intensive", "recession")
    _EIG_TOL = 1e-9  # порог для сравнения Re(λ) с нулём

    def __init__(self, params: dict[str, float] | None = None) -> None:
        """
        Args:
            params: Параметры модели (см. DEFAULT_PARAMS). Отсутствующие
                    ключи берутся из DEFAULT_PARAMS.
        """
        self.params: dict[str, float] = {**DEFAULT_PARAMS, **(params or {})}

    # ==================================================================
    # Публичный интерфейс
    # ==================================================================

    def analyze(
        self,
        A_star: float,
        phase: str = "intensive",
        k_star_method: str = "analytic",
    ) -> StabilityResult:
        """
        Полный анализ устойчивости: k*, матрица Якоби, собственные значения,
        вердикт.

        Args:
            A_star:        Уровень TFP, при котором фиксируется точка (обычно
                            берётся из симулированной траектории на нужный год).
            phase:         'extensive' | 'intensive' | 'recession'.
            k_star_method: 'analytic' (замкнутая формула, по умолчанию) или
                            'root_scalar' (численно, scipy.optimize.root_scalar).
                            Независимо от выбора, второй метод всегда
                            выполняется дополнительно как кросс-проверка.

        Returns:
            StabilityResult.

        Raises:
            ValueError: при недопустимой фазе или A_star <= 0.
        """
        if phase not in self._VALID_PHASES:
            raise ValueError(
                f"phase должен быть одним из {self._VALID_PHASES}, получено {phase!r}"
            )
        if A_star <= 0:
            raise ValueError(f"A_star должен быть > 0, получено {A_star}")

        k_star = self.find_k_star(A_star, phase=phase, method=k_star_method)

        # Кросс-проверка независимым численным методом (scipy.optimize.root_scalar)
        other_method = "root_scalar" if k_star_method == "analytic" else "analytic"
        try:
            k_star_cross = self.find_k_star(A_star, phase=phase, method=other_method)
            root_scalar_converged = True
        except (ValueError, RuntimeError):
            k_star_cross = None
            root_scalar_converged = False

        k_star_analytic = k_star if k_star_method == "analytic" else k_star_cross
        k_star_numeric   = k_star_cross if k_star_method == "analytic" else k_star

        J = self.jacobian(k_star, A_star, phase=phase)
        eigenvalues = np.linalg.eigvals(J)
        trace = float(np.trace(J))
        determinant = float(np.linalg.det(J))

        stability_type, verdict = self._classify(eigenvalues, phase)

        if k_star_numeric is not None:
            verdict += (
                f" Кросс-проверка scipy.optimize.root_scalar: "
                f"k*≈{k_star_numeric:.6f} "
                f"(аналитически k*≈{k_star_analytic:.6f})."
            )

        return StabilityResult(
            k_star=float(k_star),
            k_star_root_scalar=(
                float(k_star_numeric) if k_star_numeric is not None else None
            ),
            root_scalar_converged=root_scalar_converged,
            A_star=float(A_star),
            phase=phase,
            params_snapshot=dict(self.params),
            jacobian=J,
            eigenvalues=list(eigenvalues),
            trace=trace,
            determinant=determinant,
            stability_type=stability_type,
            verdict=verdict,
        )

    def find_k_star(
        self,
        A_star: float,
        phase: str = "intensive",
        method: str = "analytic",
    ) -> float:
        """
        Находит стационарную точку k*, при которой dk/dt = 0.

        Args:
            A_star: Уровень TFP (заморожен на время анализа).
            phase:  'extensive' | 'intensive' | 'recession'.
            method: 'analytic' — замкнутая формула
                    k* = (база)^(1/(1-α));
                    'root_scalar' — численно через
                    ``scipy.optimize.root_scalar`` (метод Брента, bracket
                    строится вокруг аналитической оценки).

        Returns:
            k* > 0.

        Raises:
            ValueError:  недопустимые входные данные, phase/method,
                         либо α = 1 (аналитическая формула не определена),
                         либо база степени <= 0.
            RuntimeError: численный солвер root_scalar не сошёлся.
        """
        if phase not in self._VALID_PHASES:
            raise ValueError(
                f"phase должен быть одним из {self._VALID_PHASES}, получено {phase!r}"
            )
        if A_star <= 0:
            raise ValueError(f"A_star должен быть > 0, получено {A_star}")

        p = self.params
        alpha = p["alpha"]

        if method == "analytic":
            base = self._k_star_base(A_star, phase, p)
            if base <= 0:
                raise ValueError(
                    f"База степени в k* = base^(1/(1-α)) <= 0 (base={base}); "
                    "стационарная точка не существует при данных параметрах."
                )
            if abs(1.0 - alpha) < 1e-12:
                raise ValueError(
                    "alpha == 1 — аналитическая формула k* не определена "
                    "(деление на (1-α)=0). Используйте method='root_scalar'."
                )
            return float(base ** (1.0 / (1.0 - alpha)))

        elif method == "root_scalar":
            return self._k_star_root_scalar(A_star, phase, p)

        else:
            raise ValueError(
                f"method должен быть 'analytic' или 'root_scalar', получено {method!r}"
            )

    def jacobian(
        self,
        k_star: float,
        A_star: float,
        phase: str = "intensive",
    ) -> "np.ndarray":
        """
        Строит матрицу Якоби линеаризованной системы (dk/dt, dA/dt)
        в точке (k*, A*) для заданной фазы.

        J = [[ ∂(dk/dt)/∂k,  ∂(dk/dt)/∂A ],
             [ ∂(dA/dt)/∂k,  ∂(dA/dt)/∂A ]]

        Args:
            k_star, A_star: Точка линеаризации.
            phase:          'extensive' | 'intensive' | 'recession'.

        Returns:
            numpy.ndarray, shape (2, 2).
        """
        if phase not in self._VALID_PHASES:
            raise ValueError(
                f"phase должен быть одним из {self._VALID_PHASES}, получено {phase!r}"
            )
        _, df_dk, df_dA, dA_dA = self._drift_and_partials(k_star, A_star, phase, self.params)
        # ∂(dA/dt)/∂k = 0 во всех фазах (A не зависит от k)
        return np.array([[df_dk, df_dA], [0.0, dA_dA]], dtype=float)

    # ==================================================================
    # Внутренняя механика
    # ==================================================================

    @staticmethod
    def _k_star_base(A_star: float, phase: str, p: dict[str, float]) -> float:
        """База степени в замкнутой формуле k* = base^(1/(1-alpha))."""
        s = p["s"]
        delta = p["delta"]
        n = p["n"]
        if phase == "extensive":
            return s * A_star / (delta + n)
        elif phase == "intensive":
            gamma = p["gamma"]
            return s * (A_star ** gamma) / (delta + n)
        else:  # recession
            s_rec = s * p["s_recession_factor"]
            dep_rec = delta * p["delta_recession_factor"] + n
            return s_rec / dep_rec

    @staticmethod
    def _drift_and_partials(
        k: float, A: float, phase: str, p: dict[str, float],
    ) -> tuple[float, float, float, float]:
        """
        Вычисляет (dk/dt, ∂(dk/dt)/∂k, ∂(dk/dt)/∂A, ∂(dA/dt)/∂A) для фазы.

        Реализация независима от ``EconomicEngine`` (не импортирует и не
        вызывает его методы), но следует тем же уравнениям фаз роста —
        единый источник истины: docstring ``EconomicEngine._phase_equations``.
        """
        alpha = p["alpha"]
        delta = p["delta"]
        n = p["n"]
        s = p["s"]

        if phase == "extensive":
            f      = s * A * k ** alpha - (delta + n) * k
            df_dk  = alpha * s * A * k ** (alpha - 1.0) - (delta + n)
            df_dA  = s * k ** alpha
            dA_dA  = 0.0

        elif phase == "intensive":
            gamma = p["gamma"]
            r     = p["r"]
            f      = s * (A ** gamma) * k ** alpha - (delta + n) * k
            df_dk  = alpha * s * (A ** gamma) * k ** (alpha - 1.0) - (delta + n)
            df_dA  = gamma * s * (A ** (gamma - 1.0)) * k ** alpha
            dA_dA  = r

        else:  # recession
            s_rec   = s * p["s_recession_factor"]
            dep_rec = delta * p["delta_recession_factor"] + n
            f      = s_rec * k ** alpha - dep_rec * k
            df_dk  = alpha * s_rec * k ** (alpha - 1.0) - dep_rec
            df_dA  = 0.0
            dA_dA  = -p["A_decay_recession"]

        return f, df_dk, df_dA, dA_dA

    def _k_star_root_scalar(self, A_star: float, phase: str, p: dict[str, float]) -> float:
        """
        Численно находит k* методом Брента (``scipy.optimize.root_scalar``).

        Строит вилку (bracket) вокруг аналитической оценки (если она
        доступна) либо на широком диапазоне [1e-8, 1e8] как fallback.

        Raises:
            RuntimeError: если солвер не нашёл смену знака / не сошёлся.
        """
        def f(k: float) -> float:
            return self._drift_and_partials(k, A_star, phase, p)[0]

        # Пытаемся построить вилку вокруг аналитической оценки —
        # значительно ускоряет и стабилизирует брентовский поиск.
        try:
            base = self._k_star_base(A_star, phase, p)
            alpha = p["alpha"]
            if base > 0 and abs(1.0 - alpha) > 1e-12:
                guess = base ** (1.0 / (1.0 - alpha))
                lo, hi = max(guess * 1e-4, 1e-10), guess * 1e4
            else:
                lo, hi = 1e-8, 1e8
        except Exception:
            lo, hi = 1e-8, 1e8

        f_lo, f_hi = f(lo), f(hi)
        if not (math.isfinite(f_lo) and math.isfinite(f_hi)):
            raise RuntimeError("dk/dt не определена (NaN/Inf) на границах вилки поиска.")

        # Если знак не меняется на исходной вилке — расширяем её логарифмически
        expand = 0
        while f_lo * f_hi > 0 and expand < 12:
            lo /= 10.0
            hi *= 10.0
            f_lo, f_hi = f(lo), f(hi)
            expand += 1

        if f_lo * f_hi > 0:
            raise RuntimeError(
                "Не удалось найти вилку со сменой знака dk/dt для "
                "scipy.optimize.root_scalar — стационарная точка, "
                "вероятно, не существует при данных параметрах."
            )

        sol = root_scalar(f, bracket=(lo, hi), method="brentq", xtol=1e-12, rtol=1e-12)
        if not sol.converged:
            raise RuntimeError(f"root_scalar не сошёлся: {sol.flag}")
        return float(sol.root)

    def _classify(
        self, eigenvalues: "np.ndarray", phase: str,
    ) -> tuple[str, str]:
        """
        Классифицирует тип устойчивости по знакам Re(λ1), Re(λ2).

        Returns:
            (stability_type, verdict) — короткая метка и развёрнутое
            текстовое объяснение для UI.
        """
        tol = self._EIG_TOL
        re = [complex(ev).real for ev in eigenvalues]
        im = [complex(ev).imag for ev in eigenvalues]
        has_imag = any(abs(i) > tol for i in im)

        lam_str = ", ".join(
            f"λ{i+1}={ev.real:+.5f}{'' if abs(ev.imag) < tol else f'{ev.imag:+.5f}i'}"
            for i, ev in enumerate(eigenvalues)
        )

        neg = [r < -tol for r in re]
        pos = [r > tol for r in re]

        if all(neg):
            stability_type = "Устойчивый фокус (аттрактор)" if has_imag else "Устойчивый узел (аттрактор)"
            verdict = (
                f"Оба собственных значения имеют отрицательную вещественную "
                f"часть ({lam_str}) — по критерию Ляпунова стационарная "
                f"точка асимптотически устойчива: малые отклонения k от k* "
                f"затухают со временем."
            )
        elif all(pos):
            stability_type = "Неустойчивый фокус (источник)" if has_imag else "Неустойчивый узел (источник)"
            verdict = (
                f"Оба собственных значения имеют положительную вещественную "
                f"часть ({lam_str}) — точка неустойчива: любое отклонение "
                f"от k* нарастает во времени."
            )
        elif any(neg) and any(pos):
            stability_type = "Седловая точка"
            verdict = (
                f"Собственные значения разных знаков ({lam_str}) — "
                f"седловая точка. Экономическая интерпретация: капитал k "
                f"локально устойчив и подстраивается к k*(A) (соответствующее "
                f"λ<0), тогда как технологии A в интенсивной фазе растут "
                f"структурно (λ=r>0) — глобального стационарного состояния "
                f"системы не существует, есть лишь «движущаяся» квази-точка "
                f"равновесия по k при растущем A."
            )
        else:
            stability_type = "Вырожденный / нейтральный случай (Re(λ)≈0)"
            verdict = (
                f"Хотя бы одно собственное значение близко к нулю "
                f"({lam_str}) — линейный анализ не даёт однозначного "
                f"вывода об устойчивости, нужен анализ высших порядков "
                f"(например, членов второго порядка) или численное "
                f"моделирование окрестности точки."
            )

        return stability_type, verdict


# ---------------------------------------------------------------------------
# 2. DataProvider
# ---------------------------------------------------------------------------

class DataProvider:
    """
    Загружает макроэкономические данные из World Bank API и преобразует их
    в параметры модели EconomicEngine.

    Ответственность:
    - Единственный класс, который знает о World Bank API.
    - Не знает о Streamlit, EconomicEngine, ScenarioManager.
    - Возвращает чистые dict / DataFrame — без UI-зависимостей.

    Кэширование:
    - ``fetch_indicator`` кэшируется через ``@st.cache_data`` (TTL = 1 час).
    - ``fetch_country_data`` также декорирован ``@st.cache_data``.
    - Оба метода сделаны ``@staticmethod`` / функциями-обёртками,
      чтобы Streamlit мог их кэшировать (``self`` не сериализуется).

    Стратегия fallback (приоритет):
    1. Данные World Bank API (последние 5 лет, среднее).
    2. Страновые константы из ``COUNTRY_FALLBACK`` (5 стран).
    3. Глобальные умолчания ``GLOBAL_FALLBACK``.

    Usage::

        data = DataProvider.fetch_country_data("KAZ")
        # → {"s": 0.212, "n": 0.0098, "gdp": 220.4e9,
        #    "source": "api", "country": "KAZ"}

        params = {**DEFAULT_PARAMS, "s": data["s"], "n": data["n"]}
        df = engine.solve(k0=2.0, A0=1.0, params=params)
    """

    # ------------------------------------------------------------------
    # Константы API
    # ------------------------------------------------------------------

    WB_BASE  = "https://api.worldbank.org/v2"
    TIMEOUT  = 10          # секунд на один HTTP-запрос
    AVG_YEARS = 5          # сколько последних лет усреднять
    CURRENT_YEAR = 2023    # последний год с полными данными World Bank

    # World Bank indicator codes
    _IND_SAVINGS = "NY.GNS.ICTR.ZS"    # Gross savings (% of GDP)
    _IND_POP_GROWTH = "SP.POP.GROW"     # Population growth (annual %)
    _IND_GDP = "NY.GDP.MKTP.CD"         # GDP (current USD)

    # ------------------------------------------------------------------
    # Страновые fallback-константы
    # Источники: World Bank Data, IMF WEO 2023, усреднение 2018-2023
    # ------------------------------------------------------------------

    COUNTRY_FALLBACK: dict[str, dict[str, float]] = {
        # Казахстан — нефтяная экономика, высокие сбережения
        "KAZ": {"s": 0.212, "n": 0.0098, "gdp": 220.4e9},
        # США — развитая экономика, низкий рост населения
        "USA": {"s": 0.183, "n": 0.0058, "gdp": 25.46e12},
        # Китай — высокие сбережения, замедляющийся рост населения
        "CHN": {"s": 0.449, "n": 0.0031, "gdp": 17.96e12},
        # Германия — высокие сбережения, убыль населения
        "DEU": {"s": 0.283, "n": -0.0011, "gdp": 4.08e12},
        # Сингапур — рекордно высокие сбережения, маленькое население
        "SGP": {"s": 0.482, "n": 0.0107, "gdp": 0.467e12},
    }

    # Глобальные умолчания (last resort)
    GLOBAL_FALLBACK: dict[str, float] = {
        "s":   0.22,
        "n":   0.01,
        "gdp": 1.0e12,
    }

    # ------------------------------------------------------------------
    # Публичный интерфейс
    # ------------------------------------------------------------------

    @staticmethod
    @st.cache_data(ttl=3600, show_spinner=False)
    def fetch_country_data(country: str) -> dict[str, Any]:
        """
        Основной публичный метод. Возвращает параметры страны для EconomicEngine.

        Алгоритм:
        1. Запрашивает три индикатора через World Bank API.
        2. Усредняет последние ``AVG_YEARS`` лет для каждого.
        3. При любой ошибке (timeout, пустые данные, неизвестная страна)
           возвращает страновые константы или глобальные умолчания.

        Args:
            country: ISO 3166-1 alpha-3 код страны (например, "KAZ", "USA").
                     Регистр не важен — нормализуется внутри.

        Returns:
            Словарь::

                {
                    "s":       float,   # норма сбережений (доля, 0–1)
                    "n":       float,   # темп роста населения (доля, напр. 0.01)
                    "gdp":     float,   # ВВП в текущих USD
                    "source":  str,     # "api" | "country_fallback" | "global_fallback"
                    "country": str,     # нормализованный код страны
                    "errors":  list[str],  # список предупреждений / ошибок
                }

        Note:
            Декорирован ``@st.cache_data(ttl=3600)`` — результат кэшируется
            на 1 час в рамках Streamlit-сессии.
        """
        code = country.strip().upper()
        errors: list[str] = []

        raw = DataProvider._fetch_all_indicators(code, errors)

        if raw is None:
            # Полная неудача API → страновой fallback
            return DataProvider._build_fallback_response(code, errors)

        result = DataProvider._aggregate(raw, code, errors)
        return result

    @staticmethod
    @st.cache_data(ttl=3600, show_spinner=False)
    def fetch_indicator_series(
        country: str,
        indicator: str,
        start: int | None = None,
        end:   int | None = None,
    ) -> pd.DataFrame:
        """
        Загружает полный временной ряд одного индикатора World Bank.

        Используется в DataExplorer вкладке (UI), не в EconomicEngine.

        Args:
            country:   ISO-3 код страны.
            indicator: Код индикатора (напр. "NY.GDP.MKTP.CD").
            start:     Начальный год (по умолчанию ``CURRENT_YEAR - 20``).
            end:       Конечный год (по умолчанию ``CURRENT_YEAR``).

        Returns:
            DataFrame с колонками: ``year`` (int), ``value`` (float).
            При ошибке — пустой DataFrame с теми же колонками.
        """
        code  = country.strip().upper()
        start = start or (DataProvider.CURRENT_YEAR - 20)
        end   = end   or DataProvider.CURRENT_YEAR

        df = DataProvider._request_indicator(code, indicator, start, end)
        if df is None:
            _log.warning(
                "fetch_indicator_series: нет данных country=%s indicator=%s",
                code, indicator,
            )
            return pd.DataFrame(columns=["year", "value"])
        return df

    # ------------------------------------------------------------------
    # Внутренние методы — HTTP-слой
    # ------------------------------------------------------------------

    @staticmethod
    def _fetch_all_indicators(
        country: str,
        errors: list[str],
    ) -> dict[str, pd.DataFrame] | None:
        """
        Запрашивает все три индикатора и возвращает словарь DataFrame-ов.

        Args:
            country: Нормализованный ISO-3 код.
            errors:  Список для накопления предупреждений (изменяется in-place).

        Returns:
            ``{"savings": df, "pop_growth": df, "gdp": df}``
            или ``None``, если все три запроса провалились.
        """
        end   = DataProvider.CURRENT_YEAR
        start = end - DataProvider.AVG_YEARS - 2   # +2 года запаса на пропуски

        mapping = {
            "savings":    DataProvider._IND_SAVINGS,
            "pop_growth": DataProvider._IND_POP_GROWTH,
            "gdp":        DataProvider._IND_GDP,
        }

        result: dict[str, pd.DataFrame] = {}
        for key, indicator in mapping.items():
            df = DataProvider._request_indicator(country, indicator, start, end)
            if df is None or df.empty:
                errors.append(f"Нет данных: country={country} indicator={indicator}")
                _log.warning("Нет данных: country=%s indicator=%s", country, indicator)
            else:
                result[key] = df

        if not result:
            _log.error("Все запросы к API провалились для country=%s", country)
            return None

        return result

    @staticmethod
    def _request_indicator(
        country: str,
        indicator: str,
        start: int,
        end: int,
    ) -> pd.DataFrame | None:
        """
        Делает один HTTP-запрос к World Bank API.

        Обрабатывает:
        - ``requests.Timeout``         → логирует, возвращает None
        - ``requests.ConnectionError`` → логирует, возвращает None
        - HTTP 4xx / 5xx               → логирует, возвращает None
        - Пустой ответ / неверный JSON → логирует, возвращает None
        - Невалидный код страны        → World Bank вернёт пустой массив data,
                                         что ловится как «нет данных»

        Args:
            country:   ISO-3 код страны.
            indicator: Код индикатора.
            start:     Начальный год.
            end:       Конечный год.

        Returns:
            DataFrame (year: int, value: float) или None при любой ошибке.
        """
        url = (
            f"{DataProvider.WB_BASE}/country/{country}/indicator/{indicator}"
            f"?format=json&date={start}:{end}&per_page=50&mrv={DataProvider.AVG_YEARS + 2}"
        )
        _log.debug("GET %s", url)

        try:
            resp = requests.get(url, timeout=DataProvider.TIMEOUT)
            resp.raise_for_status()

        except requests.Timeout:
            _log.warning("Timeout: %s", url)
            return None

        except requests.ConnectionError as exc:
            _log.warning("Ошибка соединения: %s — %s", url, exc)
            return None

        except requests.HTTPError as exc:
            _log.warning("HTTP ошибка: %s — %s", url, exc)
            return None

        except requests.RequestException as exc:
            _log.error("Неожиданная ошибка requests: %s — %s", url, exc)
            return None

        # --- Разбор JSON ---
        try:
            raw = resp.json()
        except ValueError as exc:
            _log.warning("Невалидный JSON от %s: %s", url, exc)
            return None

        # World Bank возвращает список из двух элементов: [meta, data]
        if not isinstance(raw, list) or len(raw) < 2:
            _log.warning("Неожиданная структура ответа от %s", url)
            return None

        data_block = raw[1]
        if not data_block:
            # Пустой массив данных — чаще всего неверный код страны
            _log.warning(
                "Пустой массив данных: country=%s indicator=%s. "
                "Возможно, неверный ISO-3 код страны.",
                country, indicator,
            )
            return None

        records = []
        for item in data_block:
            if item.get("value") is None:
                continue
            try:
                records.append({
                    "year":  int(item["date"]),
                    "value": float(item["value"]),
                })
            except (KeyError, ValueError, TypeError) as exc:
                _log.debug("Пропущена запись: %s — %s", item, exc)

        if not records:
            _log.warning(
                "Все значения NULL: country=%s indicator=%s",
                country, indicator,
            )
            return None

        df = pd.DataFrame(records).sort_values("year", ascending=True).reset_index(drop=True)
        return df

    # ------------------------------------------------------------------
    # Внутренние методы — агрегация
    # ------------------------------------------------------------------

    @staticmethod
    def _aggregate(
        raw: dict[str, pd.DataFrame],
        country: str,
        errors: list[str],
    ) -> dict[str, Any]:
        """
        Усредняет последние ``AVG_YEARS`` лет по каждому индикатору
        и собирает итоговый словарь параметров.

        Если какой-то индикатор отсутствует или все значения NaN —
        используется страновой или глобальный fallback для этого параметра.

        Args:
            raw:     Словарь DataFrame-ов из ``_fetch_all_indicators``.
            country: Нормализованный ISO-3 код.
            errors:  Список предупреждений (изменяется in-place).

        Returns:
            Финальный словарь параметров (см. ``fetch_country_data``).
        """
        fb = DataProvider.COUNTRY_FALLBACK.get(country, {})
        global_fb = DataProvider.GLOBAL_FALLBACK

        def mean_last_n(df: pd.DataFrame | None, n: int = DataProvider.AVG_YEARS) -> float | None:
            """Среднее по последним n ненулевым строкам."""
            if df is None or df.empty:
                return None
            tail = df.nlargest(n, "year")["value"].dropna()
            return float(tail.mean()) if not tail.empty else None

        # Норма сбережений: % → доля (делим на 100)
        s_pct = mean_last_n(raw.get("savings"))
        if s_pct is not None and s_pct > 0:
            s = s_pct / 100.0
        else:
            s = fb.get("s", global_fb["s"])
            errors.append(f"savings fallback для {country}: s={s:.4f}")
            _log.info("savings fallback: country=%s s=%.4f", country, s)

        # Темп роста населения: % → доля
        n_pct = mean_last_n(raw.get("pop_growth"))
        if n_pct is not None:
            n = n_pct / 100.0
        else:
            n = fb.get("n", global_fb["n"])
            errors.append(f"pop_growth fallback для {country}: n={n:.4f}")
            _log.info("pop_growth fallback: country=%s n=%.4f", country, n)

        # ВВП: текущие USD (без преобразования)
        gdp = mean_last_n(raw.get("gdp"))
        if gdp is None or gdp <= 0:
            gdp = fb.get("gdp", global_fb["gdp"])
            errors.append(f"gdp fallback для {country}: gdp={gdp:.3e}")
            _log.info("gdp fallback: country=%s gdp=%.3e", country, gdp)

        source = "api" if not errors else "api_partial"

        _log.info(
            "fetch_country_data OK: country=%s s=%.4f n=%.4f gdp=%.3e source=%s",
            country, s, n, gdp, source,
        )

        return {
            "s":       s,
            "n":       n,
            "gdp":     gdp,
            "source":  source,
            "country": country,
            "errors":  errors,
        }

    @staticmethod
    def _build_fallback_response(
        country: str,
        errors: list[str],
    ) -> dict[str, Any]:
        """
        Строит ответ целиком из fallback-констант (когда API полностью недоступен).

        Args:
            country: Нормализованный ISO-3 код.
            errors:  Список уже накопленных ошибок.

        Returns:
            Финальный словарь параметров с ``source = "country_fallback"``
            или ``"global_fallback"``.
        """
        if country in DataProvider.COUNTRY_FALLBACK:
            fb = DataProvider.COUNTRY_FALLBACK[country]
            source = "country_fallback"
            _log.warning(
                "API недоступен — country_fallback: country=%s s=%.4f n=%.4f gdp=%.3e",
                country, fb["s"], fb["n"], fb["gdp"],
            )
        else:
            fb = DataProvider.GLOBAL_FALLBACK
            source = "global_fallback"
            errors.append(
                f"Неизвестная страна «{country}» и API недоступен — "
                f"используются глобальные умолчания."
            )
            _log.error(
                "Неизвестная страна и API недоступен: country=%s → global_fallback",
                country,
            )

        return {
            "s":       fb["s"],
            "n":       fb["n"],
            "gdp":     fb["gdp"],
            "source":  source,
            "country": country,
            "errors":  errors,
        }


# ---------------------------------------------------------------------------
# 2.5 ModelCalibrator — автокалибровка (α, r, γ, s) под данные World Bank
# ---------------------------------------------------------------------------

@dataclass
class CalibrationResult:
    """
    Результат автокалибровки ModelCalibrator.

    Attributes:
        alpha, r, gamma, s: Откалиброванные параметры модели.
        r2:        Коэффициент детерминации между нормированным реальным
                   и смоделированным рядом ВВП на душу населения.
        rmse:      Корень из MSE (loss в оптимальной точке).
        n_points:  Число точек реального ряда, использованных для калибровки.
        method:    Метод оптимизации, фактически давший лучший результат.
        success:   Флаг успешной сходимости scipy.optimize.minimize.
        message:   Сообщение solver'а scipy.
        country:   ISO-3 код страны.
        indicator: Код индикатора World Bank, использованный как Y_real(t).
        years:     Список годов реального ряда.
        y_real_norm, y_sim_norm: Нормированные (к первому году) ряды
                   Y_real(t) и Y_sim(t, θ) — для последующей визуализации.
    """
    alpha: float
    r: float
    gamma: float
    s: float
    r2: float
    rmse: float
    n_points: int
    method: str
    success: bool
    message: str
    country: str
    indicator: str
    years: list[int] = field(default_factory=list)
    y_real_norm: list[float] = field(default_factory=list)
    y_sim_norm: list[float] = field(default_factory=list)


@dataclass
class BacktestResult:
    """
    Результат бэктеста (out-of-sample прогноза) от ``ModelCalibrator.backtest()``.

    В отличие от ``CalibrationResult`` (калибровка "в лоб" по всему периоду
    сразу, in-sample), здесь модель калибруется ТОЛЬКО на обучающей части
    ряда (``train_years``), а затем прогнозируется вперёд на отложенный
    период (``test_years``), который модель при калибровке не видела.
    Это честная проверка прогностической способности модели, а не только
    способности описывать уже известное прошлое.

    Attributes:
        country, indicator: Страна и индикатор World Bank.
        holdout_years:  Число последних лет, отложенных под прогноз.
        train_years:    Годы обучающей выборки (использованы в minimize()).
        test_years:     Годы отложенной выборки (только для сравнения,
                        НЕ участвовали в подборе θ).
        years_full:     Все годы (train + test).
        y_real_full:    Реальный ряд (норм. к первому году train) на всём периоде.
        y_sim_full:     Прогноз калиброванной модели на всём периоде
                        (та же нормировка) — первая часть до holdout — это
                        "подгонка", последние holdout_years точек — это
                        собственно прогноз (forecast).
        alpha, r, gamma, s: Параметры, откалиброванные ТОЛЬКО на train_years.
        method:         Метод оптимизации.
        success:        Флаг сходимости.
        r2_train:       R² на обучающей выборке (in-sample качество подгонки).
        rmse_train:     RMSE на обучающей выборке.
        rmse_test:      RMSE на отложенной выборке (out-of-sample) — главная
                        метрика качества ПРОГНОЗА.
        mape_test:      Средняя абсолютная процентная ошибка (%) на
                        отложенной выборке — интерпретируемая метрика
                        ("прогноз в среднем ошибается на X%").
    """
    country: str
    indicator: str
    holdout_years: int
    train_years: list[int]
    test_years: list[int]
    years_full: list[int]
    y_real_full: list[float]
    y_sim_full: list[float]
    alpha: float
    r: float
    gamma: float
    s: float
    method: str
    success: bool
    r2_train: float
    rmse_train: float
    rmse_test: float
    mape_test: float


class ModelCalibrator:
    """
    Автокалибровка несущих параметров модели (α, r, γ, s) под реальный
    исторический ряд ВВП на душу населения из World Bank (через DataProvider).

    Независим от Streamlit и от остальных orchestration-классов — принимает
    на вход EconomicEngine и код страны, возвращает CalibrationResult.

    Математика
    ----------
    Функция потерь — MSE между нормированным (к первому наблюдению) реальным
    рядом Y_real(t) и нормированным смоделированным рядом Y_sim(t, θ),
    где θ = [alpha, r, gamma, s]::

        L(θ) = (1/N) * Σ (Y_real(t_i) - Y_sim(t_i, θ))²

    Y_sim(t, θ) получается интегрированием EconomicEngine с параметрами
    θ (остальные параметры — фиксированы, берутся из DEFAULT_PARAMS
    с возможным переопределением, например n — темпом роста населения
    из DataProvider) и последующим вызовом compute_gdp(), колонка "y"
    (ВВП на работника).

    Оптимизация — scipy.optimize.minimize с жёсткими границами:
        alpha ∈ [0.1, 0.6]
        r     ∈ [0.001, 0.1]
        gamma ∈ [0.1, 0.9]
        s     ∈ [0.05, 0.5]

    По умолчанию используется L-BFGS-B; если он не сошёлся — автоматически
    делается попытка Nelder-Mead из той же начальной точки, и выбирается
    результат с меньшим значением функции потерь.

    Usage::

        engine = EconomicEngine()
        calibrator = ModelCalibrator(engine=engine, country="KAZ", k0=2.0, A0=1.0)
        result = calibrator.fit()
        print(result.alpha, result.r, result.gamma, result.s, result.r2)
    """

    # Границы параметров — жёстко заданы согласно ТЗ
    PARAM_NAMES: tuple[str, ...] = ("alpha", "r", "gamma", "s")
    BOUNDS: list[tuple[float, float]] = [
        (0.1, 0.6),     # alpha
        (0.001, 0.1),   # r
        (0.1, 0.9),     # gamma
        (0.05, 0.5),    # s
    ]

    # Индикатор World Bank по умолчанию: ВВП на душу населения,
    # постоянные цены 2015 г. (устраняет эффект инфляции при нормировке)
    DEFAULT_INDICATOR = "NY.GDP.PCAP.KD"

    def __init__(
        self,
        engine: EconomicEngine,
        country: str,
        k0: float = 2.0,
        A0: float = 1.0,
        fixed_params: dict[str, float] | None = None,
        indicator: str = DEFAULT_INDICATOR,
        start_year: int | None = None,
        end_year: int | None = None,
    ) -> None:
        """
        Args:
            engine:       EconomicEngine, используемый как шаблон (t_span
                          и dt переопределяются под длину реального ряда;
                          сам объект engine не мутируется — под капотом
                          создаётся отдельный EconomicEngine на каждый вызов).
            country:      ISO-3 код страны (напр. "KAZ").
            k0, A0:       Начальные условия для симуляции (капитал и TFP).
            fixed_params: Параметры, которые НЕ калибруются (delta, n,
                          A_critical и т.д.). Отсутствующие ключи берутся
                          из DEFAULT_PARAMS.
            indicator:    Код индикатора World Bank для Y_real(t).
            start_year, end_year: Диапазон лет для fetch_indicator_series
                          (по умолчанию — последние 20 лет).
        """
        self.engine = engine
        self.country = country.strip().upper()
        self.k0 = float(k0)
        self.A0 = float(A0)
        self.fixed_params: dict[str, float] = {**DEFAULT_PARAMS, **(fixed_params or {})}
        self.indicator = indicator
        self.start_year = start_year
        self.end_year = end_year

        self._years: list[int] = []
        self._y_real_norm: "np.ndarray | None" = None

    # ==================================================================
    # Публичный интерфейс
    # ==================================================================

    def fit(
        self,
        x0: list[float] | None = None,
        method: str = "L-BFGS-B",
    ) -> CalibrationResult:
        """
        Запускает автокалибровку (in-sample — по всему доступному периоду).

        Args:
            x0:     Начальное приближение [alpha, r, gamma, s]. По умолчанию
                    берётся из fixed_params (DEFAULT_PARAMS), обрезанное
                    по границам.
            method: "L-BFGS-B" | "Nelder-Mead". При отсутствии сходимости
                    автоматически подключается второй метод как fallback.

        Returns:
            CalibrationResult с откалиброванными параметрами и R².

        Raises:
            ValueError: если реальный ряд World Bank недоступен или
                        слишком короток (< 4 точек) для осмысленной
                        калибровки.
        """
        self._years, self._y_real_norm = self._load_real_series()

        best, used_method = self._optimize_theta(x0, method)
        alpha, r, gamma, s = (float(v) for v in best.x)

        y_sim_norm = self._simulate_normalized(best.x)
        if y_sim_norm is None:
            y_sim_norm = np.full_like(self._y_real_norm, np.nan)

        r2 = self._r_squared(self._y_real_norm, y_sim_norm)
        rmse = float(np.sqrt(max(best.fun, 0.0)))

        return CalibrationResult(
            alpha=alpha, r=r, gamma=gamma, s=s,
            r2=r2, rmse=rmse,
            n_points=len(self._years),
            method=used_method,
            success=bool(best.success),
            message=str(best.message),
            country=self.country,
            indicator=self.indicator,
            years=list(self._years),
            y_real_norm=self._y_real_norm.tolist(),
            y_sim_norm=y_sim_norm.tolist(),
        )

    def backtest(
        self,
        holdout_years: int = 5,
        x0: list[float] | None = None,
        method: str = "L-BFGS-B",
    ) -> BacktestResult:
        """
        Бэктест "на всякий случай": честная out-of-sample проверка
        прогностической способности модели.

        В отличие от ``fit()`` (калибровка сразу по всему периоду —
        in-sample, модель "видит" всё, что потом же и описывает), здесь:

        1. Из полного ряда World Bank откладываются последние
           ``holdout_years`` лет — модель их НЕ видит при калибровке.
        2. ``scipy.optimize.minimize`` подбирает θ=[α,r,γ,s] ТОЛЬКО по
           оставшейся (обучающей) части ряда.
        3. С этой же θ модель интегрируется ВПЕРЁД через весь горизонт,
           включая отложенный период.
        4. Прогноз на отложенном периоде сравнивается с реальными
           (невиданными при калибровке) данными — RMSE и MAPE.

        Если RMSE/MAPE на отложенном периоде существенно хуже, чем R²
        на обучающей выборке, — это явный сигнал переобучения
        (модель хорошо описывает прошлое, но плохо предсказывает будущее).

        Args:
            holdout_years: Число последних лет ряда, отложенных под
                           прогноз (не участвуют в оптимизации). ≥ 1.
            x0:            Начальное приближение [alpha, r, gamma, s].
            method:        "L-BFGS-B" | "Nelder-Mead" (с fallback,
                           как в ``fit()``).

        Returns:
            BacktestResult.

        Raises:
            ValueError: если данных недостаточно (после вычета holdout
                        должно остаться ≥ 4 точек для калибровки), или
                        holdout_years < 1.
        """
        if holdout_years < 1:
            raise ValueError(f"holdout_years должен быть >= 1, получено {holdout_years}")

        years_full, y_real_full = self._load_real_series()
        n_total = len(years_full)
        n_train = n_total - holdout_years

        if n_train < 4:
            raise ValueError(
                f"Недостаточно данных для бэктеста: всего точек={n_total}, "
                f"holdout_years={holdout_years} → в обучающей выборке "
                f"осталось бы {n_train} (< 4). Уменьшите holdout_years."
            )

        train_years = years_full[:n_train]
        train_y = y_real_full[:n_train]
        test_years = years_full[n_train:]

        # --- Шаг 1-2: калибровка ТОЛЬКО на обучающей выборке ---
        self._years = train_years
        self._y_real_norm = train_y

        best, used_method = self._optimize_theta(x0, method)
        alpha, r, gamma, s = (float(v) for v in best.x)

        y_sim_train = self._simulate_normalized(best.x)
        if y_sim_train is None:
            y_sim_train = np.full(n_train, np.nan)

        r2_train = self._r_squared(np.asarray(train_y, dtype=float), y_sim_train)
        rmse_train = float(np.sqrt(max(best.fun, 0.0)))

        # --- Шаг 3: интегрируем ту же калиброванную θ через ВЕСЬ горизонт
        #     (включая отложенный период — модель прогнозирует вперёд) ---
        self._years = years_full
        y_sim_full = self._simulate_normalized(best.x)
        if y_sim_full is None:
            y_sim_full = np.full(n_total, np.nan)

        # --- Шаг 4: сравниваем прогноз с реальностью на отложенном периоде ---
        y_sim_test = np.asarray(y_sim_full[n_train:], dtype=float)
        y_real_test = np.asarray(y_real_full[n_train:], dtype=float)
        valid = np.isfinite(y_sim_test)

        if np.any(valid):
            resid_test = y_real_test[valid] - y_sim_test[valid]
            rmse_test = float(np.sqrt(np.mean(resid_test ** 2)))
            mape_test = float(
                np.mean(np.abs(resid_test) / np.abs(y_real_test[valid])) * 100.0
            )
        else:
            rmse_test = float("nan")
            mape_test = float("nan")

        return BacktestResult(
            country=self.country,
            indicator=self.indicator,
            holdout_years=holdout_years,
            train_years=list(train_years),
            test_years=list(test_years),
            years_full=list(years_full),
            y_real_full=[float(v) for v in y_real_full],
            y_sim_full=[float(v) for v in np.asarray(y_sim_full, dtype=float)],
            alpha=alpha, r=r, gamma=gamma, s=s,
            method=used_method,
            success=bool(best.success),
            r2_train=r2_train,
            rmse_train=rmse_train,
            rmse_test=rmse_test,
            mape_test=mape_test,
        )

    def _optimize_theta(
        self,
        x0: list[float] | None,
        method: str,
    ) -> tuple[Any, str]:
        """
        Общая логика оптимизации, разделяемая ``fit()`` и ``backtest()``:
        запускает ``scipy.optimize.minimize`` с заданным методом, при
        несходимости — fallback на второй метод. Использует
        ``self._loss`` / ``self._years`` / ``self._y_real_norm``, которые
        должны быть выставлены ДО вызова (на train или на полный ряд).

        Returns:
            (best_optimize_result, used_method_label)
        """
        if x0 is None:
            x0 = [
                self.fixed_params["alpha"],
                self.fixed_params["r"],
                self.fixed_params["gamma"],
                self.fixed_params["s"],
            ]
        x0 = [float(np.clip(v, lo, hi)) for v, (lo, hi) in zip(x0, self.BOUNDS)]

        primary = minimize(
            self._loss,
            x0=np.array(x0, dtype=float),
            method=method,
            bounds=self.BOUNDS,
            options={"maxiter": 200},
        )
        best = primary
        used_method = method

        if not primary.success:
            fallback_method = "Nelder-Mead" if method != "Nelder-Mead" else "L-BFGS-B"
            fallback = minimize(
                self._loss,
                x0=np.array(x0, dtype=float),
                method=fallback_method,
                bounds=self.BOUNDS,
                options={"maxiter": 400, "xatol": 1e-6, "fatol": 1e-9}
                if fallback_method == "Nelder-Mead" else {"maxiter": 200},
            )
            if fallback.fun < best.fun:
                best = fallback
                used_method = f"{fallback_method} (fallback)"

        return best, used_method

    # ==================================================================
    # Загрузка и подготовка реального ряда
    # ==================================================================

    def _load_real_series(self) -> tuple[list[int], "np.ndarray"]:
        """
        Загружает реальный ряд ВВП на душу населения из DataProvider
        и нормирует его к первому наблюдению: Y_real_norm(t) = Y(t)/Y(t0).

        Returns:
            (years, y_real_norm)

        Raises:
            ValueError: если данных недостаточно (< 4 точек).
        """
        df = DataProvider.fetch_indicator_series(
            self.country, self.indicator,
            start=self.start_year, end=self.end_year,
        )
        if df is None or df.empty or len(df) < 4:
            raise ValueError(
                f"Недостаточно данных World Bank для калибровки: "
                f"country={self.country} indicator={self.indicator} "
                f"(получено {0 if df is None else len(df)} точек, нужно ≥4)."
            )

        df = df.sort_values("year").reset_index(drop=True)
        years = [int(y) for y in df["year"].tolist()]
        values = df["value"].to_numpy(dtype=float)

        if values[0] <= 0:
            raise ValueError(
                f"Первое значение ряда ({values[0]}) <= 0 — нормировка невозможна."
            )

        y_real_norm = values / values[0]
        return years, y_real_norm

    # ==================================================================
    # Симуляция модели при заданных θ
    # ==================================================================

    def _simulate_normalized(self, theta: "np.ndarray") -> "np.ndarray | None":
        """
        Прогоняет EconomicEngine с параметрами θ и возвращает нормированный
        (к первому году) смоделированный ряд Y_sim(t) той же длины, что
        и реальный ряд.

        Использует шаг dt=1.0 год, чтобы точки симуляции совпадали
        с годовыми точками реального ряда World Bank.

        Args:
            theta: [alpha, r, gamma, s].

        Returns:
            np.ndarray нормированного ряда или None, если симуляция
            завершилась ошибкой / привела к нефизичным значениям.
        """
        alpha, r, gamma, s = (float(v) for v in theta)
        n_points = len(self._years)
        t_end = float(n_points - 1)

        local_engine = EconomicEngine(t_span=(0.0, max(t_end, 1.0)), dt=1.0)
        params = {**self.fixed_params, "alpha": alpha, "r": r, "gamma": gamma, "s": s}

        try:
            df = local_engine.solve(k0=self.k0, A0=self.A0, params=params, shocks=[])
            df = local_engine.compute_gdp(df, params)
        except (ValueError, RuntimeError):
            return None

        y_sim = df["y"].to_numpy(dtype=float)
        if len(y_sim) < n_points:
            return None
        y_sim = y_sim[:n_points]

        if y_sim[0] <= 0 or not np.all(np.isfinite(y_sim)):
            return None

        return y_sim / y_sim[0]

    # ==================================================================
    # Функция потерь и метрики
    # ==================================================================

    def _loss(self, theta: "np.ndarray") -> float:
        """
        MSE между нормированным реальным и смоделированным рядом.

        Возвращает большое конечное число (штраф) вместо исключения,
        если симуляция при данных θ нефизична — это удерживает
        scipy.optimize внутри допустимой области, не прерывая оптимизацию.
        """
        y_sim_norm = self._simulate_normalized(theta)
        if y_sim_norm is None:
            return 1e6

        resid = self._y_real_norm - y_sim_norm
        mse = float(np.mean(resid ** 2))
        return mse if math.isfinite(mse) else 1e6

    @staticmethod
    def _r_squared(y_real: "np.ndarray", y_sim: "np.ndarray") -> float:
        """
        Коэффициент детерминации R² = 1 - SS_res / SS_tot.

        Returns:
            R² (может быть отрицательным, если модель хуже среднего;
            NaN, если реальный ряд константен — SS_tot = 0).
        """
        if np.any(~np.isfinite(y_sim)):
            return float("nan")
        ss_res = float(np.sum((y_real - y_sim) ** 2))
        ss_tot = float(np.sum((y_real - np.mean(y_real)) ** 2))
        if ss_tot < 1e-12:
            return float("nan")
        return 1.0 - ss_res / ss_tot


# ---------------------------------------------------------------------------
# 3. ScenarioManager
# ---------------------------------------------------------------------------

class ScenarioManager:
    """
    Управляет сценариями Digital Twin Economics.

    Три уровня воздействия на модель (применяются последовательно в ``build_params``):
    1. **Reforms**   — постоянные структурные изменения параметров с t >= apply_at_t.
    2. **Modifiers** — масштабирующие/аддитивные ползунки политики (сбережения, R&D).
    3. **Shock**     — временной кризис с собственными param_overrides и принудительной фазой.

    Принципы изоляции:
    - ``ScenarioManager`` ничего не знает о математике EconomicEngine.
    - ``EconomicEngine`` ничего не знает о реформах и модификаторах.
    - Единственный канал передачи — ``build_params()`` и ``get_shocks()``.
    - Все «магические числа» шоков живут в ``SHOCK_CATALOG``, не внутри engine.

    Usage::

        sm = ScenarioManager(base_params=DEFAULT_PARAMS)

        # Политика
        sm.set_modifier("savings_stimulus", param="s", mode="add", value=0.05)
        sm.set_modifier("rd_investment",    param="r", mode="scale", value=1.4)

        # Реформа
        sm.add_reform("Пенсионная реформа", param="n", delta=-0.003, apply_at_t=5.0)

        # Шок
        sm.set_shock("capital_crisis", start_t=8.0, duration=3.0)

        # Финальная сборка params для EconomicEngine
        params = sm.build_params(current_t=10.0)
        shocks = sm.get_shocks()
        df = engine.solve(k0=2.0, A0=1.0, params=params, shocks=shocks)
    """

    def __init__(self, base_params: dict[str, float] | None = None) -> None:
        """
        Args:
            base_params: Базовый словарь параметров (DEFAULT_PARAMS или из DataProvider).
                         Если не передан — используется DEFAULT_PARAMS.
        """
        # Импортируем здесь, чтобы не создавать циклическую зависимость
        self._base: dict[str, float] = dict(base_params or DEFAULT_PARAMS)

        self._reforms:   list[Reform]             = []
        self._modifiers: dict[str, Modifier]      = {}   # name → Modifier
        self._shock_key: str                      = "no_shock"
        self._shock_start_t: float                = 0.0
        self._shock_duration: float               = 5.0

    # ==================================================================
    # Reforms
    # ==================================================================

    def add_reform(
        self,
        name: str,
        param: str,
        delta: float,
        apply_at_t: float,
        description: str = "",
    ) -> Reform:
        """
        Добавляет структурную реформу.

        Если реформа с таким именем уже существует — заменяет её.

        Args:
            name:        Уникальное название реформы.
            param:       Ключ параметра в DEFAULT_PARAMS (напр. "s", "delta", "r").
            delta:       Абсолютное изменение: итог = base + delta.
            apply_at_t:  Год начала действия реформы.
            description: Описание экономического смысла (для UI).

        Returns:
            Созданный объект Reform.

        Raises:
            KeyError: если ``param`` не является ключом DEFAULT_PARAMS.
        """
        if param not in DEFAULT_PARAMS:
            raise KeyError(
                f"Параметр {param!r} не найден в DEFAULT_PARAMS. "
                f"Допустимые: {sorted(DEFAULT_PARAMS)}"
            )
        self._reforms = [r for r in self._reforms if r.name != name]  # upsert
        reform = Reform(name=name, param=param, delta=delta,
                        apply_at_t=apply_at_t, description=description)
        self._reforms.append(reform)
        return reform

    def remove_reform(self, name: str) -> None:
        """Удаляет реформу по имени. Молчит, если реформы нет."""
        self._reforms = [r for r in self._reforms if r.name != name]

    def apply_reforms(
        self,
        params: dict[str, float],
        current_t: float,
    ) -> dict[str, float]:
        """
        Применяет все активные реформы к словарю params.

        Реформа считается активной, если ``current_t >= reform.apply_at_t``.
        Несколько реформ на один param суммируются.

        Args:
            params:    Входной словарь (не мутируется — возвращается копия).
            current_t: Текущий момент времени симуляции.

        Returns:
            Новый словарь params с применёнными реформами.
        """
        result = dict(params)
        for reform in self._reforms:
            if current_t >= reform.apply_at_t:
                old = result.get(reform.param, 0.0)
                result[reform.param] = old + reform.delta
        return result

    # ==================================================================
    # Modifiers (ползунки политики)
    # ==================================================================

    def set_modifier(
        self,
        name: str,
        param: str,
        mode: str,
        value: float,
        label: str = "",
    ) -> Modifier:
        """
        Устанавливает (или обновляет) внешний модификатор параметра.

        Модификаторы применяются после реформ и до сборки финального params.
        Типичное использование — ползунки Streamlit (savings_stimulus, rd_investment).

        Args:
            name:  Уникальный идентификатор (напр. "savings_stimulus").
            param: Ключ в DEFAULT_PARAMS.
            mode:  "add" (param += value) или "scale" (param *= value).
            value: Числовое значение.
            label: Отображаемое название (для UI).

        Returns:
            Созданный/обновлённый Modifier.

        Raises:
            KeyError:   если ``param`` не в DEFAULT_PARAMS.
            ValueError: если ``mode`` не "add" / "scale".
        """
        if param not in DEFAULT_PARAMS:
            raise KeyError(f"Параметр {param!r} не найден в DEFAULT_PARAMS.")
        mod = Modifier(name=name, param=param, mode=mode, value=value, label=label or name)
        self._modifiers[name] = mod
        return mod

    def remove_modifier(self, name: str) -> None:
        """Удаляет модификатор по имени. Молчит, если нет."""
        self._modifiers.pop(name, None)

    def _apply_modifiers(self, params: dict[str, float]) -> dict[str, float]:
        """
        Применяет все зарегистрированные модификаторы к params.

        Порядок применения — порядок добавления (dict insertion order, Python 3.7+).
        При mode="scale" значение умножается, при mode="add" — прибавляется.

        Args:
            params: Входной словарь (не мутируется).

        Returns:
            Новый словарь с применёнными модификаторами.
        """
        result = dict(params)
        for mod in self._modifiers.values():
            old = result.get(mod.param, 0.0)
            if mod.mode == "add":
                result[mod.param] = old + mod.value
            else:  # "scale"
                result[mod.param] = old * mod.value
        return result

    # ==================================================================
    # Shock
    # ==================================================================

    def set_shock(
        self,
        shock_key: str,
        start_t: float,
        duration: float,
    ) -> None:
        """
        Устанавливает активный шок по ключу из SHOCK_CATALOG.

        За один раз активен только один шок. Повторный вызов заменяет предыдущий.
        Для отключения шока передайте shock_key="no_shock".

        Args:
            shock_key: Ключ шока из SHOCK_CATALOG ("no_shock", "capital_crisis",
                       "tech_stagnation").
            start_t:   Год начала шока.
            duration:  Длительность шока в годах.

        Raises:
            KeyError: если shock_key не найден в SHOCK_CATALOG.
        """
        if shock_key not in SHOCK_CATALOG:
            raise KeyError(
                f"Шок {shock_key!r} не найден в SHOCK_CATALOG. "
                f"Доступные: {list(SHOCK_CATALOG)}"
            )
        self._shock_key      = shock_key
        self._shock_start_t  = float(start_t)
        self._shock_duration = float(duration)

    def get_shocks(self) -> list[Shock]:
        """
        Возвращает список Shock для передачи в EconomicEngine.

        Если активный шок — "no_shock", возвращает пустой список.
        Иначе — список из одного Shock с интервалом (start_t, duration).

        Returns:
            list[Shock] — передаётся напрямую в ``engine.solve(shocks=...)``.
        """
        if self._shock_key == "no_shock":
            return []

        template = SHOCK_CATALOG[self._shock_key]
        shock = Shock(
            shock_type=ShockType.SUPPLY,   # тип используется только для совместимости
            magnitude=1.0,                 # величина не нужна — логика в param_overrides
            duration=self._shock_duration,
            start_t=self._shock_start_t,
            label=template.label,
        )
        return [shock]

    def get_active_shock_template(self) -> ShockTemplate:
        """Возвращает активный ShockTemplate (может быть "no_shock")."""
        return SHOCK_CATALOG[self._shock_key]

    # ==================================================================
    # build_params — главный публичный метод
    # ==================================================================

    def build_params(self, current_t: float = 0.0) -> dict[str, float]:
        """
        Собирает финальный словарь параметров для EconomicEngine.

        Порядок применения слоёв:
        1. base_params                   (DEFAULT_PARAMS или из DataProvider)
        2. apply_reforms(current_t)      (постоянные структурные изменения)
        3. _apply_modifiers()            (ползунки политики)
        4. shock param_overrides         (только если шок активен в момент current_t)

        Слой 4 применяется только если текущее время попадает в интервал шока,
        чтобы params отражали текущее состояние (используется в get_state).
        Для solve_ivp шок передаётся отдельно через get_shocks() и обрабатывается
        внутри _rhs по интервалу времени.

        Args:
            current_t: Текущий момент времени (год). Влияет на:
                       - какие реформы уже активны;
                       - активен ли шок прямо сейчас (для отображения в UI).

        Returns:
            Словарь параметров, готовый к передаче в ``engine.solve(params=...)``.
        """
        # Слой 1: база
        params = dict(self._base)

        # Слой 2: реформы
        params = self.apply_reforms(params, current_t)

        # Слой 3: модификаторы (ползунки)
        params = self._apply_modifiers(params)

        # Слой 4: param_overrides активного шока (только в интервале шока)
        template = SHOCK_CATALOG[self._shock_key]
        shock_active = (
            self._shock_key != "no_shock"
            and self._shock_start_t <= current_t < self._shock_start_t + self._shock_duration
        )
        if shock_active and template.param_overrides:
            params = {**params, **template.param_overrides}

        return params

    # ==================================================================
    # Обновление базы
    # ==================================================================

    def update_base(self, new_base: dict[str, float]) -> None:
        """
        Обновляет базовый словарь параметров (напр. после загрузки из DataProvider).

        Только известные ключи (из DEFAULT_PARAMS) принимаются.
        Остальные молча игнорируются.

        Args:
            new_base: Новый словарь параметров (может быть частичным).
        """
        for key, val in new_base.items():
            if key in DEFAULT_PARAMS:
                self._base[key] = val

    # ==================================================================
    # Сброс и экспорт
    # ==================================================================

    def reset(self) -> None:
        """Полный сброс: реформы, модификаторы, шок → начальное состояние."""
        self._reforms.clear()
        self._modifiers.clear()
        self._shock_key      = "no_shock"
        self._shock_start_t  = 0.0
        self._shock_duration = 5.0

    def summary(self) -> pd.DataFrame:
        """
        Сводная таблица всех зарегистрированных событий (для UI).

        Returns:
            DataFrame с колонками: event_type, name, param, value, start_t, notes.
        """
        rows: list[dict[str, Any]] = []

        for r in self._reforms:
            rows.append({
                "event_type": "Реформа",
                "name":       r.name,
                "param":      r.param,
                "value":      f"Δ{r.delta:+.4f}",
                "start_t":    r.apply_at_t,
                "notes":      r.description,
            })

        for m in self._modifiers.values():
            sign = "+" if m.mode == "add" else "×"
            rows.append({
                "event_type": "Модификатор",
                "name":       m.label,
                "param":      m.param,
                "value":      f"{sign}{m.value:.4f}",
                "start_t":    0.0,
                "notes":      f"mode={m.mode}",
            })

        if self._shock_key != "no_shock":
            t = SHOCK_CATALOG[self._shock_key]
            rows.append({
                "event_type": "Шок",
                "name":       t.label,
                "param":      "multiple",
                "value":      f"t=[{self._shock_start_t}, "
                              f"{self._shock_start_t + self._shock_duration}]",
                "start_t":    self._shock_start_t,
                "notes":      t.description,
            })

        return pd.DataFrame(rows) if rows else pd.DataFrame(
            columns=["event_type", "name", "param", "value", "start_t", "notes"]
        )

    # ==================================================================
    # Совместимость со старым API (get_active)
    # ==================================================================

    def get_active(self) -> tuple[list[Shock], list[Reform]]:
        """
        Обратная совместимость со старым API.

        Returns:
            (shocks, reforms) — для передачи в engine.solve().
        """
        return self.get_shocks(), list(self._reforms)


# ---------------------------------------------------------------------------
# 3.5 AdvancedShockManager — гладкие (логистические) и циклические шоки
# ---------------------------------------------------------------------------

class ShockProfile(str, Enum):
    """
    Временной профиль продвинутого шока.

    STEP:      Базовое ступенчатое поведение (см. ``ScenarioManager`` +
               ``SHOCK_CATALOG``) — сохраняется на 100% без изменений.
    LOGISTIC:  Плавный сигмоидальный переход параметра к новому уровню
               (реформа / постепенно проявляющийся кризис).
    CYCLICAL:  Гармонические (синусоидальные) колебания параметра —
               бизнес-циклы (волны Жюгляра/Кондратьева).
    """
    STEP      = "step"
    LOGISTIC  = "logistic"
    CYCLICAL  = "cyclical"


@dataclass
class AdvancedShockConfig:
    """
    Конфигурация одного продвинутого (гладкого/циклического) шока.

    Attributes:
        profile:   Временной профиль (ShockProfile).
        param:     Модулируемый параметр модели (напр. "s" или "delta").
        start_t:   Год начала действия ("центр" сигмоиды для LOGISTIC,
                    начало отсчёта фазы синуса для CYCLICAL).
        delta:     [LOGISTIC] Итоговое изменение параметра при t → +∞
                    (Δs в формуле s(t) = s0 + Δs/(1+e^{-k_speed(t-t0)})).
        k_speed:   [LOGISTIC] Скорость перехода (крутизна сигмоиды).
        amplitude: [CYCLICAL] Относительная амплитуда колебаний A_amp
                    (доля от базового значения параметра).
        period:    [CYCLICAL] Длина цикла T_cycle (лет).
        label:     Человекочитаемое описание (для UI/summary).
    """
    profile: ShockProfile
    param: str
    start_t: float
    delta: float = 0.0
    k_speed: float = 1.0
    amplitude: float = 0.1
    period: float = 8.0
    label: str = ""

    def __post_init__(self) -> None:
        if not self.label:
            if self.profile == ShockProfile.LOGISTIC:
                self.label = (
                    f"Логистический шок {self.param}: "
                    f"Δ={self.delta:+.3f}, k={self.k_speed:.2f}, t0={self.start_t:.1f}"
                )
            elif self.profile == ShockProfile.CYCLICAL:
                self.label = (
                    f"Циклический шок {self.param}: "
                    f"A={self.amplitude:.2f}, T={self.period:.1f}, t0={self.start_t:.1f}"
                )
            else:
                self.label = f"Ступенчатый шок {self.param}"


class AdvancedShockManager:
    """
    Обогащает ``ScenarioManager`` поддержкой гладких (логистических) и
    циклических шоков, дополняя базовое ступенчатое поведение, а не
    заменяя его: при профиле ``ShockProfile.STEP`` (или при отсутствии
    конфигурации) метод ``as_dynamic_params_fn()`` возвращает ``None``,
    и ``EconomicEngine.solve()`` работает ИДЕНТИЧНО тому, как работал
    до добавления этого модуля — 100% обратная совместимость.

    Математика
    ----------
    Логистический (плавный) переход::

        s(t) = s0 + Δs / (1 + e^{-k_speed·(t - t_start)})

    Циклический (гармонический) шок — бизнес-цикл::

        s(t) = s0 · (1 + A_amp · sin(2π/T_cycle · (t - t_start)))   для t >= t_start
        s(t) = s0                                                     для t <  t_start

    где s0 — текущее («базовое», уже включающее реформы/модификаторы)
    значение модулируемого параметра из ``params`` в момент оценки.

    Композиция, а не наследование
    -------------------------------
    Класс НЕ наследует ``ScenarioManager`` (чтобы не дублировать и не
    переопределять его публичный API), а работает как независимый
    plug-in: собранную функцию ``as_dynamic_params_fn()`` нужно передать
    в ``EconomicEngine.solve(..., dynamic_params=...)`` рядом с
    ``params=sm.build_params()`` и ``shocks=sm.get_shocks()``.

    Usage::

        asm = AdvancedShockManager()
        asm.set_logistic_shock(param="s", start_t=5.0, delta=-0.08, k_speed=1.2)
        # или: asm.set_cyclical_shock(param="s", start_t=0.0, amplitude=0.15, period=8.0)

        df = engine.solve(
            k0=2.0, A0=1.0,
            params=sm.build_params(), shocks=sm.get_shocks(),
            dynamic_params=asm.as_dynamic_params_fn(),
        )
    """

    # Разумные физические границы для клиппинга — не позволяют
    # плавным/циклическим колебаниям увести параметр в нефизичную область
    # (напр. отрицательная норма сбережений).
    PARAM_BOUNDS: dict[str, tuple[float, float]] = {
        "s":     (0.0, 0.95),
        "delta": (0.0, 0.95),
        "r":     (0.0, 0.50),
        "alpha": (0.05, 0.95),
        "gamma": (0.05, 0.95),
        "n":     (-0.05, 0.10),
    }

    def __init__(self) -> None:
        self._config: AdvancedShockConfig | None = None

    # ==================================================================
    # Конфигурация
    # ==================================================================

    def set_logistic_shock(
        self,
        param: str,
        start_t: float,
        delta: float,
        k_speed: float,
        label: str = "",
    ) -> AdvancedShockConfig:
        """
        Настраивает плавный (логистический) шок/реформу.

        Args:
            param:   Модулируемый параметр (должен быть в DEFAULT_PARAMS).
            start_t: Год «центра» перехода (t0 в формуле сигмоиды).
            delta:   Итоговое изменение параметра (Δ) при t → +∞.
            k_speed: Скорость перехода (крутизна сигмоиды, > 0 — быстрее).
            label:   Опциональное описание для UI.

        Returns:
            Созданный AdvancedShockConfig.

        Raises:
            KeyError: если param не в DEFAULT_PARAMS.
        """
        if param not in DEFAULT_PARAMS:
            raise KeyError(f"Параметр {param!r} не найден в DEFAULT_PARAMS.")
        cfg = AdvancedShockConfig(
            profile=ShockProfile.LOGISTIC, param=param,
            start_t=float(start_t), delta=float(delta), k_speed=float(k_speed),
            label=label,
        )
        self._config = cfg
        return cfg

    def set_cyclical_shock(
        self,
        param: str,
        start_t: float,
        amplitude: float,
        period: float,
        label: str = "",
    ) -> AdvancedShockConfig:
        """
        Настраивает циклический (гармонический) шок — бизнес-цикл.

        Args:
            param:     Модулируемый параметр (должен быть в DEFAULT_PARAMS).
            start_t:   Год начала колебаний (до этого момента — s(t)=s0).
            amplitude: Относительная амплитуда A_amp (доля от s0, обычно 0..0.5).
            period:    Длина цикла T_cycle в годах (> 0).
            label:     Опциональное описание для UI.

        Returns:
            Созданный AdvancedShockConfig.

        Raises:
            KeyError:   если param не в DEFAULT_PARAMS.
            ValueError: если period <= 0.
        """
        if param not in DEFAULT_PARAMS:
            raise KeyError(f"Параметр {param!r} не найден в DEFAULT_PARAMS.")
        if period <= 0:
            raise ValueError(f"period должен быть > 0, получено {period}")
        cfg = AdvancedShockConfig(
            profile=ShockProfile.CYCLICAL, param=param,
            start_t=float(start_t), amplitude=float(amplitude), period=float(period),
            label=label,
        )
        self._config = cfg
        return cfg

    def clear(self) -> None:
        """Отключает продвинутый шок (возврат к базовому ступенчатому поведению)."""
        self._config = None

    def get_config(self) -> AdvancedShockConfig | None:
        """Возвращает активную конфигурацию (или None, если не задана)."""
        return self._config

    # ==================================================================
    # Вычисление и интеграция с EconomicEngine
    # ==================================================================

    def value_at(self, t: float, base_value: float) -> float:
        """
        Вычисляет значение модулируемого параметра в момент t.

        Args:
            t:          Момент времени (год).
            base_value: Базовое (немодулированное) значение параметра s0.

        Returns:
            Значение параметра с учётом активного профиля (обрезанное по
            PARAM_BOUNDS). Если конфиг не задан или профиль STEP —
            возвращает ``base_value`` без изменений.
        """
        cfg = self._config
        if cfg is None or cfg.profile == ShockProfile.STEP:
            return base_value

        if cfg.profile == ShockProfile.LOGISTIC:
            # s(t) = s0 + Δs / (1 + e^{-k_speed*(t - t0)})
            x = -cfg.k_speed * (t - cfg.start_t)
            x = max(min(x, 700.0), -700.0)  # защита от overflow в math.exp
            sigmoid = 1.0 / (1.0 + math.exp(x))
            value = base_value + cfg.delta * sigmoid

        elif cfg.profile == ShockProfile.CYCLICAL:
            if t < cfg.start_t:
                value = base_value
            else:
                omega = 2.0 * math.pi / cfg.period
                value = base_value * (1.0 + cfg.amplitude * math.sin(omega * (t - cfg.start_t)))
        else:
            value = base_value

        lo, hi = self.PARAM_BOUNDS.get(cfg.param, (None, None))
        if lo is not None:
            value = max(lo, min(hi, value))
        return value

    def as_dynamic_params_fn(
        self,
    ) -> Callable[[float, dict[str, float]], dict[str, float]] | None:
        """
        Собирает функцию ``f(t, params) -> {param: value}`` для передачи в
        ``EconomicEngine.solve(dynamic_params=...)``.

        Returns:
            Функция, либо ``None``, если активного гладкого/циклического
            шока нет (профиль STEP или конфиг не задан) — в этом случае
            вызывающий код может просто не передавать ``dynamic_params``,
            и поведение EconomicEngine останется на 100% прежним.
        """
        cfg = self._config
        if cfg is None or cfg.profile == ShockProfile.STEP:
            return None

        param = cfg.param

        def _dynamic_params(t: float, p: dict[str, float]) -> dict[str, float]:
            base_value = p.get(param, DEFAULT_PARAMS.get(param, 0.0))
            return {param: self.value_at(t, base_value)}

        return _dynamic_params

    def profile_series(
        self, t_grid: "np.ndarray", base_value: float,
    ) -> "np.ndarray":
        """
        Вспомогательный метод для UI/визуализации: считает s(t) (или
        любого другого модулируемого параметра) на сетке t_grid, не
        запуская интегрирование.

        Args:
            t_grid:     Массив моментов времени.
            base_value: Базовое значение параметра (s0).

        Returns:
            numpy.ndarray той же длины, что и t_grid.
        """
        return np.array([self.value_at(float(t), base_value) for t in t_grid])


# ---------------------------------------------------------------------------
# 4. PlotBuilder
# ---------------------------------------------------------------------------

# Цветовая палитра — единственное место, где живут цвета
_PALETTE = {
    "baseline":    "#00d4ff",   # голубой  — базовый сценарий
    "reform":      "#6bcb77",   # зелёный  — сценарий с реформами
    "shock":       "#ff6b6b",   # красный  — шок / рецессия
    "attractor":   "#ffd93d",   # жёлтый   — аттрактор (устойчивое равновесие)
    "equilibrium": "#c77dff",   # фиолетовый — равновесная точка
    "phase_ext":   "#00d4ff",   # extensive
    "phase_int":   "#6bcb77",   # intensive
    "phase_rec":   "#ff6b6b",   # recession
    "grid":        "rgba(255,255,255,0.08)",
    "annotation":  "rgba(255,255,255,0.85)",
}

# Константы оформления
_FONT_TITLE  = dict(size=15, color="white")
_FONT_AXIS   = dict(size=12, color="#cccccc")
_FONT_LEGEND = dict(size=11, color="white")
_LINE_WIDTH  = 2
_MARKER_SIZE = 9


class PlotBuilder:
    """
    Строит интерактивные графики Plotly для Digital Twin Economics.

    Принципы:
    - Принимает **только** готовый ``pandas.DataFrame`` — никакой математики.
    - Все вычисления (производные, равновесие, ВВП) — на стороне EconomicEngine.
    - Колонки, которые ожидаются в DataFrame, задокументированы в каждом методе.
    - Единственная логика внутри: маппинг фаз → числа/цвета для Plotly.

    Ожидаемые колонки DataFrame из ``EconomicEngine.solve()`` + ``compute_gdp()``:
        t, k, A, phase, dk_dt, dA_dt, L, K, Y, y

    Usage::

        pb  = PlotBuilder(theme="plotly_dark")

        # Два основных требуемых графика:
        fig1 = pb.build_timeseries_plot(df_base, df_reform)
        fig2 = pb.build_phase_plot(df)

        st.plotly_chart(fig1, use_container_width=True)
        st.plotly_chart(fig2, use_container_width=True)
    """

    # Маппинг фаза → числовой код (для осей Plotly)
    PHASE_NUM  = {"extensive": 0, "intensive": 1, "recession": 2}
    PHASE_COLOR = {
        "extensive": _PALETTE["phase_ext"],
        "intensive": _PALETTE["phase_int"],
        "recession": _PALETTE["phase_rec"],
    }

    def __init__(self, theme: str = "plotly_dark") -> None:
        """
        Args:
            theme: Тема Plotly (``"plotly_dark"`` | ``"plotly_white"`` | ``"plotly"``).
        """
        self.theme = theme

    # ==================================================================
    # ① build_timeseries_plot  — ОСНОВНОЙ ТРЕБУЕМЫЙ МЕТОД
    # ==================================================================

    def build_timeseries_plot(
        self,
        df_base: pd.DataFrame,
        df_reform: pd.DataFrame,
        *,
        y_col: str = "y",
        shock_intervals: list[tuple[float, float]] | None = None,
        title: str = "ВВП на работника: Baseline vs Reforms",
    ) -> go.Figure:
        """
        Временной ряд: две кривые ВВП — baseline и с реформами.

        Ожидаемые колонки обоих DataFrame:
            t  — время (годы)
            y  — ВВП на работника (из ``engine.compute_gdp()``)
               — если колонка ``y`` отсутствует, используется ``k`` как прокси

        Дополнительно наносятся:
        - Цветные фоновые полосы для интервалов шоков (если переданы).
        - Аннотация максимального расхождения между кривыми.

        Args:
            df_base:         DataFrame базового сценария.
            df_reform:       DataFrame сценария с реформами/шоками.
            y_col:           Колонка для оси Y (по умолчанию ``"y"``).
            shock_intervals: Список кортежей ``(t_start, t_end)`` для подсветки шоков.
            title:           Заголовок графика.

        Returns:
            go.Figure с двумя кривыми, легендой, осями и (опционально) шоковыми полосами.
        """
        col_base   = y_col if y_col in df_base.columns   else "k"
        col_reform = y_col if y_col in df_reform.columns else "k"

        y_label = "ВВП на работника y" if y_col == "y" else f"{y_col}"

        fig = go.Figure()

        # --- Baseline ---
        fig.add_trace(go.Scatter(
            x=df_base["t"],
            y=df_base[col_base],
            mode="lines",
            name="Baseline",
            line=dict(color=_PALETTE["baseline"], width=_LINE_WIDTH),
            hovertemplate=(
                "<b>Baseline</b><br>"
                "t = %{x:.2f} лет<br>"
                f"{y_label} = %{{y:.4f}}<extra></extra>"
            ),
        ))

        # --- Reforms ---
        fig.add_trace(go.Scatter(
            x=df_reform["t"],
            y=df_reform[col_reform],
            mode="lines",
            name="С реформами / шоком",
            line=dict(color=_PALETTE["reform"], width=_LINE_WIDTH, dash="dash"),
            hovertemplate=(
                "<b>С реформами</b><br>"
                "t = %{x:.2f} лет<br>"
                f"{y_label} = %{{y:.4f}}<extra></extra>"
            ),
        ))

        # --- Фоновые полосы шоков ---
        if shock_intervals:
            self._add_shock_bands(fig, shock_intervals)

        # --- Аннотация максимального расхождения ---
        self._annotate_max_divergence(fig, df_base, df_reform, col_base, col_reform, y_label)

        fig.update_layout(
            template=self.theme,
            title=dict(text=title, font=_FONT_TITLE),
            xaxis=dict(
                title=dict(text="Время (годы симуляции)", font=_FONT_AXIS),
                gridcolor=_PALETTE["grid"],
                showgrid=True,
            ),
            yaxis=dict(
                title=dict(text=y_label, font=_FONT_AXIS),
                gridcolor=_PALETTE["grid"],
                showgrid=True,
            ),
            legend=dict(
                font=_FONT_LEGEND,
                bgcolor="rgba(0,0,0,0.3)",
                bordercolor="rgba(255,255,255,0.2)",
                borderwidth=1,
                x=0.01, y=0.99,
                xanchor="left", yanchor="top",
            ),
            hovermode="x unified",
            height=420,
            margin=dict(l=60, r=30, t=60, b=50),
        )
        return fig

    # ==================================================================
    # ② build_phase_plot  — ОСНОВНОЙ ТРЕБУЕМЫЙ МЕТОД
    # ==================================================================

    def build_phase_plot(
        self,
        df: pd.DataFrame,
        *,
        title: str = "Фазовое пространство: k vs dk/dt",
    ) -> go.Figure:
        """
        Фазовый портрет: ось X = k, ось Y = dk/dt.

        Визуализирует траекторию экономики в фазовом пространстве.
        Раскрашивает точки по текущей фазе роста.
        Подсвечивает аттрактор (устойчивую точку равновесия Солоу) и
        равновесную точку (k*, где dk/dt ≈ 0).

        Ожидаемые колонки DataFrame:
            k      — капиталовооружённость
            dk_dt  — производная dk/dt (из ``engine.solve()``)
            phase  — строковое название фазы

        Аттрактор и равновесие вычислены **заранее** (в engine) и переданы
        уже готовыми в DataFrame — PlotBuilder их только отображает.
        Если колонок ``k_star`` / ``attractor_k`` нет — определяет из данных.

        Args:
            df:    DataFrame из ``engine.solve()`` (обязательно с ``dk_dt``).
            title: Заголовок графика.

        Returns:
            go.Figure с траекторией, нулевой изоклиной, аттрактором, равновесием.
        """
        fig = go.Figure()

        # --- Траектория по фазам (раскраска) ---
        for phase_name, color in self.PHASE_COLOR.items():
            mask = df["phase"] == phase_name
            if not mask.any():
                continue
            sub = df[mask]
            fig.add_trace(go.Scatter(
                x=sub["k"],
                y=sub["dk_dt"],
                mode="lines+markers",
                name=f"Фаза: {phase_name}",
                line=dict(color=color, width=_LINE_WIDTH),
                marker=dict(size=4, color=color, opacity=0.7),
                hovertemplate=(
                    f"<b>Фаза: {phase_name}</b><br>"
                    "k = %{x:.4f}<br>"
                    "dk/dt = %{y:.5f}<extra></extra>"
                ),
            ))

        # --- Нулевая изоклина (dk/dt = 0) ---
        k_min_val = float(df["k"].min())
        k_max_val = float(df["k"].max())
        fig.add_hline(
            y=0.0,
            line=dict(color="rgba(255,255,255,0.35)", width=1, dash="dot"),
            annotation_text="dk/dt = 0",
            annotation_position="bottom right",
            annotation_font=dict(color="rgba(255,255,255,0.6)", size=10),
        )

        # --- Равновесная точка k* (где |dk/dt| минимален) ---
        eq_idx  = df["dk_dt"].abs().idxmin()
        eq_k    = float(df.loc[eq_idx, "k"])
        eq_dkdt = float(df.loc[eq_idx, "dk_dt"])

        fig.add_trace(go.Scatter(
            x=[eq_k],
            y=[eq_dkdt],
            mode="markers+text",
            name="Равновесие k*",
            marker=dict(
                color=_PALETTE["equilibrium"],
                size=_MARKER_SIZE + 3,
                symbol="star",
                line=dict(color="white", width=1),
            ),
            text=["k*"],
            textposition="top center",
            textfont=dict(color=_PALETTE["equilibrium"], size=11),
            hovertemplate=(
                "<b>Равновесие k*</b><br>"
                "k* = %{x:.4f}<br>"
                "dk/dt ≈ %{y:.5f}<extra></extra>"
            ),
        ))

        # --- Аттрактор (последняя точка траектории — куда система стремится) ---
        attr_k    = float(df["k"].iloc[-1])
        attr_dkdt = float(df["dk_dt"].iloc[-1])

        fig.add_trace(go.Scatter(
            x=[attr_k],
            y=[attr_dkdt],
            mode="markers+text",
            name="Аттрактор",
            marker=dict(
                color=_PALETTE["attractor"],
                size=_MARKER_SIZE + 5,
                symbol="diamond",
                line=dict(color="white", width=1.5),
            ),
            text=["Аттрактор"],
            textposition="top right",
            textfont=dict(color=_PALETTE["attractor"], size=11),
            hovertemplate=(
                "<b>Аттрактор</b><br>"
                "k = %{x:.4f}<br>"
                "dk/dt = %{y:.5f}<extra></extra>"
            ),
        ))

        # --- Вертикальная линия через аттрактор ---
        fig.add_vline(
            x=attr_k,
            line=dict(color=_PALETTE["attractor"], width=1, dash="dashdot"),
            annotation_text=f"k→{attr_k:.3f}",
            annotation_position="top",
            annotation_font=dict(color=_PALETTE["attractor"], size=10),
        )

        # --- Стрелка направления движения (первая → последняя точка) ---
        self._add_direction_arrow(fig, df)

        fig.update_layout(
            template=self.theme,
            title=dict(text=title, font=_FONT_TITLE),
            xaxis=dict(
                title=dict(text="Капиталовооружённость k", font=_FONT_AXIS),
                gridcolor=_PALETTE["grid"],
                showgrid=True,
                range=[max(0, k_min_val * 0.9), k_max_val * 1.1],
            ),
            yaxis=dict(
                title=dict(text="dk/dt (скорость изменения капитала)", font=_FONT_AXIS),
                gridcolor=_PALETTE["grid"],
                showgrid=True,
            ),
            legend=dict(
                font=_FONT_LEGEND,
                bgcolor="rgba(0,0,0,0.3)",
                bordercolor="rgba(255,255,255,0.2)",
                borderwidth=1,
                x=0.01, y=0.99,
                xanchor="left", yanchor="top",
            ),
            hovermode="closest",
            height=460,
            margin=dict(l=70, r=30, t=60, b=55),
        )
        return fig

    # ==================================================================
    # ②.5 add_monte_carlo_bands — стохастический шум / Монте-Карло
    # ==================================================================

    def add_monte_carlo_bands(
        self,
        mc: "MonteCarloResult",
        *,
        variable: str = "y",
        max_lines: int = 30,
        title: str | None = None,
    ) -> go.Figure:
        """
        Визуализирует результат ``StochasticEngine.run_monte_carlo()``:

        - До ``max_lines`` полупрозрачных тонких индивидуальных траекторий
          (не все ``n_runs``, чтобы график оставался читаемым).
        - Закрашенную область 95% доверительного интервала между
          2.5-м и 97.5-м перцентилями (``fill="tonexty"``).
        - Медианную траекторию (50-й перцентиль) поверх области.

        Args:
            mc:        MonteCarloResult из ``StochasticEngine.run_monte_carlo()``.
            variable:  ``"y"`` (ВВП на работника) или ``"k"`` (капиталовооружённость).
            max_lines: Максимальное число индивидуальных траекторий для отрисовки.
            title:     Заголовок графика (генерируется автоматически, если не задан).

        Returns:
            go.Figure с тонкими траекториями, 95% доверительной областью и медианой.

        Raises:
            ValueError: если ``variable`` не ``"y"`` и не ``"k"``.
        """
        if variable not in ("y", "k"):
            raise ValueError(f"variable должен быть 'y' или 'k', получено {variable!r}")

        paths = mc.y_paths if variable == "y" else mc.k_paths
        q = mc.quantiles[variable]
        y_label = "ВВП на работника y" if variable == "y" else "Капиталовооружённость k"

        if title is None:
            title = (
                f"Монте-Карло: {y_label}(t) — "
                f"N={mc.n_runs} прогонов, σ={mc.sigma:.3f}"
            )

        fig = go.Figure()

        # --- Индивидуальные тонкие полупрозрачные траектории ---
        n_show = min(max_lines, mc.n_runs)
        # Равномерно выбираем подвыборку индексов, а не только первые N,
        # чтобы не создавать смещение по seed-порядку генерации.
        idx = np.linspace(0, mc.n_runs - 1, n_show).round().astype(int)
        idx = np.unique(idx)
        for j, i in enumerate(idx):
            fig.add_trace(go.Scatter(
                x=mc.t,
                y=paths[i],
                mode="lines",
                line=dict(color=_PALETTE["baseline"], width=1),
                opacity=0.15,
                showlegend=False,
                hoverinfo="skip",
            ))

        # --- 95% доверительная область (fill='tonexty' между границами) ---
        fig.add_trace(go.Scatter(
            x=mc.t,
            y=q["p97_5"],
            mode="lines",
            line=dict(width=0),
            showlegend=False,
            hoverinfo="skip",
            name="97.5-й перцентиль",
        ))
        fig.add_trace(go.Scatter(
            x=mc.t,
            y=q["p2_5"],
            mode="lines",
            line=dict(width=0),
            fill="tonexty",
            fillcolor="rgba(0, 212, 255, 0.20)",
            name="95% доверительный интервал",
            hovertemplate=(
                "t = %{x:.2f} лет<br>"
                f"[2.5%, 97.5%] {y_label}<extra></extra>"
            ),
        ))

        # --- Медиана (p50) поверх области ---
        fig.add_trace(go.Scatter(
            x=mc.t,
            y=q["p50"],
            mode="lines",
            name="Медиана (p50)",
            line=dict(color=_PALETTE["reform"], width=_LINE_WIDTH),
            hovertemplate=(
                "<b>Медиана</b><br>"
                "t = %{x:.2f} лет<br>"
                f"{y_label} = %{{y:.4f}}<extra></extra>"
            ),
        ))

        fig.update_layout(
            template=self.theme,
            title=dict(text=title, font=_FONT_TITLE),
            xaxis=dict(
                title=dict(text="Время (годы симуляции)", font=_FONT_AXIS),
                gridcolor=_PALETTE["grid"],
                showgrid=True,
            ),
            yaxis=dict(
                title=dict(text=y_label, font=_FONT_AXIS),
                gridcolor=_PALETTE["grid"],
                showgrid=True,
            ),
            legend=dict(
                font=_FONT_LEGEND,
                bgcolor="rgba(0,0,0,0.3)",
                bordercolor="rgba(255,255,255,0.2)",
                borderwidth=1,
                x=0.01, y=0.99,
                xanchor="left", yanchor="top",
            ),
            hovermode="x unified",
            height=460,
            margin=dict(l=60, r=30, t=60, b=50),
        )
        return fig

    # ==================================================================
    # ②.7 build_calibration_fit_plot — реальный vs откалиброванный ряд
    # ==================================================================

    def build_calibration_fit_plot(self, result: "CalibrationResult") -> go.Figure:
        """
        Визуализирует качество подгонки ``ModelCalibrator``: нормированный
        реальный ряд World Bank Y_real(t) против нормированного
        смоделированного ряда Y_sim(t, θ*) при откалиброванных θ*.

        Args:
            result: CalibrationResult из ``ModelCalibrator.fit()``.

        Returns:
            go.Figure с двумя линиями и заливкой ошибки между ними.
        """
        years = result.years
        y_real = result.y_real_norm
        y_sim = result.y_sim_norm

        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=years, y=y_real, mode="lines+markers",
            name="Реальность (World Bank)",
            line=dict(color=_PALETTE["baseline"], width=_LINE_WIDTH),
            marker=dict(size=5),
        ))
        fig.add_trace(go.Scatter(
            x=years, y=y_sim, mode="lines+markers",
            name="Модель (откалиброванная)",
            line=dict(color=_PALETTE["reform"], width=_LINE_WIDTH, dash="dash"),
            marker=dict(size=5, symbol="diamond"),
        ))

        r2_txt = f"{result.r2:.4f}" if math.isfinite(result.r2) else "н/д"
        fig.update_layout(
            template=self.theme,
            title=dict(
                text=(
                    f"Калибровка {result.country}: реальность vs модель "
                    f"(R²={r2_txt}, метод: {result.method})"
                ),
                font=_FONT_TITLE,
            ),
            xaxis=dict(
                title=dict(text="Год", font=_FONT_AXIS),
                gridcolor=_PALETTE["grid"], showgrid=True,
            ),
            yaxis=dict(
                title=dict(text="ВВП/душу насел., норм. к первому году", font=_FONT_AXIS),
                gridcolor=_PALETTE["grid"], showgrid=True,
            ),
            legend=dict(
                font=_FONT_LEGEND,
                bgcolor="rgba(0,0,0,0.3)",
                bordercolor="rgba(255,255,255,0.2)",
                borderwidth=1,
                x=0.01, y=0.99,
                xanchor="left", yanchor="top",
            ),
            hovermode="x unified",
            height=440,
            margin=dict(l=60, r=30, t=60, b=50),
        )
        return fig

    # ==================================================================
    # ②.8 build_backtest_plot — бэктест (out-of-sample прогноз)
    # ==================================================================

    def build_backtest_plot(self, result: "BacktestResult") -> go.Figure:
        """
        Визуализирует бэктест ``ModelCalibrator.backtest()``: реальный ряд
        целиком vs прогноз модели, откалиброванной ТОЛЬКО на обучающей
        части. Отложенный (прогнозный) период выделен цветной областью —
        сразу видно, где заканчивается «память» модели и начинается
        честный прогноз вперёд.

        Args:
            result: BacktestResult из ``ModelCalibrator.backtest()``.

        Returns:
            go.Figure с реальным рядом, прогнозом и подсветкой holdout-периода.
        """
        years = result.years_full
        y_real = result.y_real_full
        y_sim = result.y_sim_full
        split_year = result.train_years[-1] if result.train_years else years[0]
        test_start = result.test_years[0] if result.test_years else split_year

        fig = go.Figure()

        # Область прогноза (holdout) — рисуется первой, чтобы быть "под" линиями
        fig.add_vrect(
            x0=test_start - 0.5,
            x1=years[-1] + 0.5,
            fillcolor="rgba(255, 165, 0, 0.14)",
            line_width=0,
            annotation_text="Прогноз (out-of-sample)",
            annotation_position="top left",
            annotation_font=dict(size=11, color="#ffbb55"),
        )

        fig.add_trace(go.Scatter(
            x=years, y=y_real, mode="lines+markers",
            name="Реальность (World Bank)",
            line=dict(color=_PALETTE["baseline"], width=_LINE_WIDTH),
            marker=dict(size=5),
        ))
        fig.add_trace(go.Scatter(
            x=years, y=y_sim, mode="lines+markers",
            name=f"Модель (калибр. по {result.train_years[0]}–{split_year})",
            line=dict(color=_PALETTE["reform"], width=_LINE_WIDTH, dash="dash"),
            marker=dict(size=5, symbol="diamond"),
        ))

        r2_txt = f"{result.r2_train:.3f}" if math.isfinite(result.r2_train) else "н/д"
        mape_txt = f"{result.mape_test:.1f}%" if math.isfinite(result.mape_test) else "н/д"
        fig.update_layout(
            template=self.theme,
            title=dict(
                text=(
                    f"Бэктест {result.country}: обучение {result.train_years[0]}–{split_year} "
                    f"→ прогноз {result.holdout_years} лет вперёд "
                    f"(R²_train={r2_txt}, MAPE_прогноза={mape_txt})"
                ),
                font=_FONT_TITLE,
            ),
            xaxis=dict(
                title=dict(text="Год", font=_FONT_AXIS),
                gridcolor=_PALETTE["grid"], showgrid=True,
            ),
            yaxis=dict(
                title=dict(text="ВВП/душу насел., норм. к первому году", font=_FONT_AXIS),
                gridcolor=_PALETTE["grid"], showgrid=True,
            ),
            legend=dict(
                font=_FONT_LEGEND,
                bgcolor="rgba(0,0,0,0.3)",
                bordercolor="rgba(255,255,255,0.2)",
                borderwidth=1,
                x=0.01, y=0.99,
                xanchor="left", yanchor="top",
            ),
            hovermode="x unified",
            height=460,
            margin=dict(l=60, r=30, t=70, b=50),
        )
        return fig

    # ==================================================================
    # ③ dashboard — комбинированный 2×2 дашборд
    # ==================================================================

    def dashboard(self, df: pd.DataFrame) -> go.Figure:
        """
        Комбинированный дашборд 2×2: k(t), A(t), y(t), фаза(t).

        Ожидаемые колонки: t, k, A, phase, y (опционально).

        Args:
            df: DataFrame из ``engine.solve()`` + ``engine.compute_gdp()``.

        Returns:
            go.Figure с четырьмя subplot'ами.
        """
        from plotly.subplots import make_subplots

        phase_numeric = df["phase"].map(self.PHASE_NUM).fillna(0)
        gdp_col = "y" if "y" in df.columns else "k"

        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=[
                "k(t) — Капиталовооружённость",
                "A(t) — Уровень технологий (TFP)",
                "y(t) — ВВП на работника",
                "Фаза роста",
            ],
            vertical_spacing=0.14,
            horizontal_spacing=0.10,
        )

        # k(t)
        fig.add_trace(go.Scatter(
            x=df["t"], y=df["k"],
            name="k", line=dict(color=_PALETTE["baseline"], width=_LINE_WIDTH),
            hovertemplate="t=%{x:.1f}<br>k=%{y:.4f}<extra>k</extra>",
        ), row=1, col=1)

        # A(t)
        fig.add_trace(go.Scatter(
            x=df["t"], y=df["A"],
            name="A (TFP)", line=dict(color=_PALETTE["shock"], width=_LINE_WIDTH),
            hovertemplate="t=%{x:.1f}<br>A=%{y:.4f}<extra>A</extra>",
        ), row=1, col=2)

        # y(t) / k(t)
        fig.add_trace(go.Scatter(
            x=df["t"], y=df[gdp_col],
            name="y (ВВП/работника)", line=dict(color=_PALETTE["reform"], width=_LINE_WIDTH),
            hovertemplate="t=%{x:.1f}<br>y=%{y:.4f}<extra>y</extra>",
        ), row=2, col=1)

        # Фаза — цветные маркеры
        phase_colors = df["phase"].map(self.PHASE_COLOR).fillna(_PALETTE["baseline"])
        fig.add_trace(go.Scatter(
            x=df["t"],
            y=phase_numeric,
            mode="lines+markers",
            name="Фаза",
            line=dict(color=_PALETTE["attractor"], width=1.5),
            marker=dict(color=list(phase_colors), size=5),
            hovertemplate="t=%{x:.1f}<br>фаза=%{text}<extra></extra>",
            text=df["phase"],
        ), row=2, col=2)

        fig.update_yaxes(
            tickvals=[0, 1, 2],
            ticktext=["extensive", "intensive", "recession"],
            row=2, col=2,
        )

        # Оси
        for r, c, xl, yl in [
            (1, 1, "Годы", "k"),
            (1, 2, "Годы", "A"),
            (2, 1, "Годы", "y"),
            (2, 2, "Годы", "Фаза"),
        ]:
            fig.update_xaxes(title_text=xl, gridcolor=_PALETTE["grid"], row=r, col=c)
            fig.update_yaxes(gridcolor=_PALETTE["grid"], row=r, col=c)

        fig.update_layout(
            template=self.theme,
            title=dict(text="Digital Twin Economics — Dashboard", font=_FONT_TITLE),
            showlegend=False,
            height=620,
            margin=dict(l=55, r=25, t=70, b=45),
        )
        return fig

    # ==================================================================
    # ④ k_chart, A_chart, gdp_chart — одиночные графики
    # ==================================================================

    def k_chart(self, df: pd.DataFrame) -> go.Figure:
        """
        График капиталовооружённости k(t).

        Колонки: t, k.
        """
        return self._single_line(
            df, x_col="t", y_col="k",
            color=_PALETTE["baseline"],
            name="k — капитал/работника",
            title="Капиталовооружённость k(t)",
            x_label="Год симуляции",
            y_label="k",
        )

    def A_chart(self, df: pd.DataFrame) -> go.Figure:
        """
        График уровня технологий A(t) / TFP.

        Колонки: t, A.
        """
        return self._single_line(
            df, x_col="t", y_col="A",
            color=_PALETTE["shock"],
            name="A — TFP",
            title="Уровень технологий A(t)",
            x_label="Год симуляции",
            y_label="A (TFP)",
        )

    def gdp_chart(self, df: pd.DataFrame) -> go.Figure:
        """
        График ВВП на работника y(t).

        Колонки: t, y (или t, k как прокси).
        """
        y_col  = "y" if "y" in df.columns else "k"
        y_label = "ВВП на работника y" if y_col == "y" else "k (прокси ВВП)"
        return self._single_line(
            df, x_col="t", y_col=y_col,
            color=_PALETTE["reform"],
            name=y_label,
            title="ВВП на работника y(t)",
            x_label="Год симуляции",
            y_label=y_label,
        )

    def phase_chart(self, df: pd.DataFrame) -> go.Figure:
        """
        График переключения фаз во времени.

        Каждый интервал раскрашивается по фазе.
        Колонки: t, phase.
        """
        phase_numeric = df["phase"].map(self.PHASE_NUM).fillna(0)
        phase_colors  = df["phase"].map(self.PHASE_COLOR).fillna(_PALETTE["baseline"])

        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=df["t"],
            y=phase_numeric,
            mode="lines+markers",
            name="Фаза роста",
            line=dict(color=_PALETTE["attractor"], width=_LINE_WIDTH),
            marker=dict(color=list(phase_colors), size=6),
            hovertemplate="t=%{x:.1f}<br>фаза=%{text}<extra></extra>",
            text=df["phase"],
        ))
        fig.update_layout(
            template=self.theme,
            title=dict(text="Смена фаз роста", font=_FONT_TITLE),
            xaxis=dict(title=dict(text="Год симуляции", font=_FONT_AXIS),
                       gridcolor=_PALETTE["grid"]),
            yaxis=dict(
                tickvals=[0, 1, 2],
                ticktext=["extensive", "intensive", "recession"],
                gridcolor=_PALETTE["grid"],
            ),
            height=300,
            margin=dict(l=60, r=25, t=50, b=45),
        )
        return fig

    # ==================================================================
    # ⑤ scenario_comparison
    # ==================================================================

    def scenario_comparison(
        self,
        dfs: dict[str, pd.DataFrame],
        variable: str = "y",
    ) -> go.Figure:
        """
        Сравнение произвольного числа сценариев по одной переменной.

        Args:
            dfs:      ``{название_сценария: DataFrame}``.
            variable: Колонка для сравнения (``"k"``, ``"A"``, ``"y"`` и т.д.).

        Returns:
            go.Figure с одной кривой на сценарий.
        """
        palette_cycle = [
            _PALETTE["baseline"], _PALETTE["reform"], _PALETTE["shock"],
            _PALETTE["attractor"], _PALETTE["equilibrium"],
            "#a8dadc", "#e63946", "#457b9d",
        ]
        fig = go.Figure()
        for i, (scenario_name, df) in enumerate(dfs.items()):
            col = variable if variable in df.columns else "k"
            fig.add_trace(go.Scatter(
                x=df["t"],
                y=df[col],
                mode="lines",
                name=scenario_name,
                line=dict(color=palette_cycle[i % len(palette_cycle)], width=_LINE_WIDTH),
                hovertemplate=(
                    f"<b>{scenario_name}</b><br>"
                    "t = %{x:.2f}<br>"
                    f"{col} = %{{y:.4f}}<extra></extra>"
                ),
            ))
        var_label = {"k": "k (капиталовооружённость)", "A": "A (TFP)",
                     "y": "y (ВВП/работника)"}.get(variable, variable)
        fig.update_layout(
            template=self.theme,
            title=dict(text=f"Сравнение сценариев — {var_label}", font=_FONT_TITLE),
            xaxis=dict(title=dict(text="Год симуляции", font=_FONT_AXIS),
                       gridcolor=_PALETTE["grid"]),
            yaxis=dict(title=dict(text=var_label, font=_FONT_AXIS),
                       gridcolor=_PALETTE["grid"]),
            legend=dict(font=_FONT_LEGEND, bgcolor="rgba(0,0,0,0.3)", borderwidth=1),
            hovermode="x unified",
            height=420,
            margin=dict(l=60, r=25, t=55, b=50),
        )
        return fig

    # ==================================================================
    # Вспомогательные (приватные) методы — только оформление
    # ==================================================================

    def _single_line(
        self,
        df: pd.DataFrame,
        x_col: str,
        y_col: str,
        color: str,
        name: str,
        title: str,
        x_label: str,
        y_label: str,
    ) -> go.Figure:
        """
        Строит простой одиночный линейный график.

        Никакой математики — только визуализация готовых колонок.
        """
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=df[x_col],
            y=df[y_col],
            mode="lines",
            name=name,
            line=dict(color=color, width=_LINE_WIDTH),
            hovertemplate=f"{x_col}=%{{x:.2f}}<br>{y_col}=%{{y:.4f}}<extra>{name}</extra>",
        ))
        fig.update_layout(
            template=self.theme,
            title=dict(text=title, font=_FONT_TITLE),
            xaxis=dict(title=dict(text=x_label, font=_FONT_AXIS), gridcolor=_PALETTE["grid"]),
            yaxis=dict(title=dict(text=y_label, font=_FONT_AXIS), gridcolor=_PALETTE["grid"]),
            legend=dict(font=_FONT_LEGEND),
            height=360,
            margin=dict(l=60, r=25, t=55, b=50),
        )
        return fig

    @staticmethod
    def _add_shock_bands(
        fig: go.Figure,
        shock_intervals: list[tuple[float, float]],
    ) -> None:
        """
        Добавляет полупрозрачные вертикальные полосы для интервалов шоков.

        Не выполняет вычислений — только рисует прямоугольники по переданным данным.

        Args:
            fig:             Figure, на которой рисуем.
            shock_intervals: Список кортежей ``(t_start, t_end)``.
        """
        for t_start, t_end in shock_intervals:
            fig.add_vrect(
                x0=t_start, x1=t_end,
                fillcolor=_PALETTE["shock"],
                opacity=0.12,
                layer="below",
                line_width=0,
                annotation_text="Шок",
                annotation_position="top left",
                annotation_font=dict(color=_PALETTE["shock"], size=10),
            )

    @staticmethod
    def _annotate_max_divergence(
        fig: go.Figure,
        df_base: pd.DataFrame,
        df_reform: pd.DataFrame,
        col_base: str,
        col_reform: str,
        y_label: str,
    ) -> None:
        """
        Находит момент максимального расхождения двух кривых и добавляет аннотацию.

        Единственная «вычислительная» операция здесь — поиск argmax разности,
        что является чисто отображательной задачей (где поставить подпись).
        Никакой экономической математики.

        Args:
            fig:       Figure для аннотации.
            df_base:   DataFrame базовой кривой.
            df_reform: DataFrame кривой с реформами.
            col_base:  Колонка Y в df_base.
            col_reform: Колонка Y в df_reform.
            y_label:   Название переменной для текста аннотации.
        """
        # Выравниваем по общей оси t через merge
        merged = df_base[["t", col_base]].merge(
            df_reform[["t", col_reform]].rename(columns={col_reform: "_reform"}),
            on="t", how="inner",
        )
        if merged.empty:
            return

        merged["_diff"] = (merged["_reform"] - merged[col_base]).abs()
        idx_max = merged["_diff"].idxmax()
        t_max   = float(merged.loc[idx_max, "t"])
        diff    = float(merged.loc[idx_max, "_diff"])
        y_val   = float(merged.loc[idx_max, col_base])

        if diff < 1e-8:
            return  # кривые совпадают — аннотация не нужна

        fig.add_annotation(
            x=t_max,
            y=y_val,
            text=f"Δ{y_label}={diff:.3f}",
            showarrow=True,
            arrowhead=2,
            arrowcolor=_PALETTE["annotation"],
            arrowsize=1,
            arrowwidth=1.5,
            font=dict(color=_PALETTE["annotation"], size=10),
            bgcolor="rgba(0,0,0,0.5)",
            bordercolor=_PALETTE["annotation"],
            borderwidth=1,
            ax=40, ay=-35,
        )

    @staticmethod
    def _add_direction_arrow(fig: go.Figure, df: pd.DataFrame) -> None:
        """
        Добавляет стрелку от начальной точки траектории к конечной
        в фазовом пространстве (k, dk/dt).

        Помогает понять направление движения системы без дополнительных подписей.
        Не выполняет математических вычислений.

        Args:
            fig: Figure фазового портрета.
            df:  DataFrame с колонками k, dk_dt.
        """
        if len(df) < 2:
            return

        x0 = float(df["k"].iloc[0])
        y0 = float(df["dk_dt"].iloc[0])
        x1 = float(df["k"].iloc[len(df) // 3])   # треть пути
        y1 = float(df["dk_dt"].iloc[len(df) // 3])

        fig.add_annotation(
            x=x1, y=y1,
            ax=x0, ay=y0,
            axref="x", ayref="y",
            xref="x", yref="y",
            showarrow=True,
            arrowhead=3,
            arrowsize=1.5,
            arrowwidth=2,
            arrowcolor="rgba(255,255,255,0.5)",
        )


# ---------------------------------------------------------------------------
# 5. AppController
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# 5. AppController
# ---------------------------------------------------------------------------

# Справочник стран: ISO-3 → (отображаемое название, k0, A0)
_COUNTRIES: dict[str, tuple[str, float, float]] = {
    "KAZ": ("🇰🇿 Казахстан",  2.0, 1.0),
    "USA": ("🇺🇸 США",         8.0, 2.5),
    "CHN": ("🇨🇳 Китай",       5.0, 1.8),
    "DEU": ("🇩🇪 Германия",    7.0, 2.2),
    "SGP": ("🇸🇬 Сингапур",    6.5, 2.0),
}

# Горизонт симуляции — фиксирован для метрик «at t=50»
_T_METRIC = 50.0
_T_END     = 55.0   # немного дольше, чтобы t=50 гарантированно было внутри


class AppController:
    """
    Orchestration-слой Streamlit UI для Digital Twin Economics.

    Единственная ответственность — связать виджеты Streamlit с методами
    EconomicEngine, ScenarioManager, DataProvider и PlotBuilder.

    Правила:
    - Никакой математики и бизнес-логики — только вызовы других классов.
    - Вся логика параметров — в ScenarioManager.build_params().
    - Все расчёты — в EconomicEngine.
    - Все графики — в PlotBuilder.
    - Session state используется исключительно для хранения результатов
      между перерисовками Streamlit.

    Структура UI:
        Sidebar  → страна, реформы (savings / R&D), шок (radio)
        Main     → метрики (t=50), вкладки Time Series / Phase Space
    """

    def __init__(self) -> None:
        self.engine   = EconomicEngine(t_span=(0.0, _T_END), dt=0.5)
        self.scenario = ScenarioManager(base_params=DEFAULT_PARAMS)
        self.plotter  = PlotBuilder(theme="plotly_dark")

    # ==================================================================
    # Точка входа
    # ==================================================================

    def run(self) -> None:
        """
        Главная точка входа. Вызывается из ``if __name__ == '__main__'``.

        Порядок работы:
        1. Конфигурация страницы.
        2. Инициализация session_state.
        3. Отрисовка sidebar (страна + сценарий).
        4. Запуск двух симуляций (baseline + сценарий).
        5. Отрисовка метрик и графиков в main window.
        """
        st.set_page_config(
            page_title="Digital Twin Economics",
            page_icon="📈",
            layout="wide",
        )
        self._init_session_state()

        # --- Sidebar возвращает все настройки пользователя ---
        ui = self._render_sidebar()

        # --- Заголовок ---
        st.title("📈 Digital Twin Economics")
        st.caption(
            f"Модель Солоу · Страна: **{_COUNTRIES[ui['country']][0]}** · "
            f"Горизонт: {_T_END:.0f} лет · Метрики при t={_T_METRIC:.0f}"
        )
        st.divider()

        # --- Запуск симуляций ---
        self._run_simulations(ui)

        # --- Main window ---
        if st.session_state["df_baseline"] is not None:
            self._render_metrics()
            self._render_charts()
            self._render_data_expander()

    # ==================================================================
    # Sidebar — возвращает dict с настройками, не хранит состояние
    # ==================================================================

    def _render_sidebar(self) -> dict[str, Any]:
        """
        Отрисовывает sidebar и возвращает словарь UI-настроек.

        Возвращает:
            {
                "country":       str,    # ISO-3
                "k0":            float,
                "A0":            float,
                "alpha":         float,  # базовое α (структурный параметр)
                "r_base":        float,  # базовое r (структурный параметр)
                "gamma":         float,  # базовое γ (структурный параметр)
                "s_base":        float,  # базовое s (структурный параметр)
                "savings_delta": float,  # Δs модификатор (поверх s_base)
                "rd_scale":      float,  # ×r модификатор (поверх r_base)
                "shock_key":     str,    # ключ из SHOCK_CATALOG
                "shock_start":   float,
                "shock_dur":     float,
                "adv_profile":   str,    # "step" | "logistic" | "cyclical"
                "adv_param":     str,    # модулируемый параметр ("s" | "delta")
                "adv_start":     float,
                "adv_delta":     float,  # [logistic]
                "adv_k_speed":   float,  # [logistic]
                "adv_amplitude": float,  # [cyclical]
                "adv_period":    float,  # [cyclical]
                "mc_enabled":    bool,   # включён ли стохастический шум (Монте-Карло)
                "mc_sigma":      float,  # волатильность σ ∈ [0.0, 0.2]
                "mc_n_runs":     int,    # число прогонов Монте-Карло
                "run_clicked":   bool,
            }

        Все возвращаемые значения — чистые Python-типы, без виджетов.
        AppController читает их и передаёт в ScenarioManager / Engine.
        """
        with st.sidebar:
            st.header("⚙️ Параметры")

            # ----------------------------------------------------------
            # Применение отложенной калибровки — ДО создания виджетов
            # sb_alpha / sb_r / sb_gamma / sb_s (иначе Streamlit запретит
            # программно менять session_state ключа уже созданного виджета
            # в рамках одного прогона скрипта).
            # ----------------------------------------------------------
            if st.session_state.get("_apply_calibration_pending"):
                _calib = st.session_state.get("calib_result")
                if _calib is not None:
                    st.session_state["sb_alpha"] = _calib.alpha
                    st.session_state["sb_r"]     = _calib.r
                    st.session_state["sb_gamma"] = _calib.gamma
                    st.session_state["sb_s"]     = _calib.s
                st.session_state["_apply_calibration_pending"] = False

            # ----------------------------------------------------------
            # Блок 1: Выбор страны
            # ----------------------------------------------------------
            st.subheader("🌍 Страна")
            country_labels = [v[0] for v in _COUNTRIES.values()]
            country_codes  = list(_COUNTRIES.keys())

            selected_label = st.selectbox(
                "Выберите страну",
                options=country_labels,
                index=0,
                key="sb_country",
                help="Реальные данные загружаются из World Bank API (fallback при недоступности).",
            )
            country = country_codes[country_labels.index(selected_label)]
            _, k0_default, A0_default = _COUNTRIES[country]

            with st.expander("Начальные условия (k₀, A₀)"):
                k0 = st.number_input(
                    "k₀ — начальная капиталовооружённость",
                    value=k0_default, min_value=0.01, step=0.1, key="sb_k0",
                )
                A0 = st.number_input(
                    "A₀ — начальный уровень TFP",
                    value=A0_default, min_value=0.01, step=0.05, key="sb_A0",
                )

            st.divider()

            # ----------------------------------------------------------
            # Блок 1.5: Структурные параметры (α, r, γ, s) + автокалибровка
            # ----------------------------------------------------------
            st.subheader("🧮 Структурные параметры (α, r, γ, s)")

            alpha_val = st.slider(
                "α — эластичность выпуска по капиталу",
                min_value=0.10, max_value=0.60,
                value=DEFAULT_PARAMS["alpha"], step=0.01,
                key="sb_alpha",
                help="Доля капитала в производственной функции Кобба–Дугласа.",
            )
            r_val = st.slider(
                "r — темп роста TFP (интенсивная фаза)",
                min_value=0.001, max_value=0.100,
                value=DEFAULT_PARAMS["r"], step=0.001,
                key="sb_r", format="%.3f",
                help="Базовый темп роста технологий до применения модификатора ×r ниже.",
            )
            gamma_val = st.slider(
                "γ — эластичность выпуска по TFP",
                min_value=0.10, max_value=0.90,
                value=DEFAULT_PARAMS["gamma"], step=0.01,
                key="sb_gamma",
                help="Эластичность капитала по TFP в интенсивной фазе роста.",
            )
            s_val = st.slider(
                "s — норма сбережений (база)",
                min_value=0.05, max_value=0.50,
                value=DEFAULT_PARAMS["s"], step=0.005,
                key="sb_s",
                help="Базовая норма сбережений до применения модификатора Δs ниже.",
            )
            st.caption(
                "Базовые структурные параметры модели. Можно задать вручную "
                "или подобрать автоматически кнопкой ниже."
            )

            with st.expander("🎯 Автокалибровка параметров (Scipy Optimize)"):
                st.caption(
                    "Подбирает α, r, γ, s методом `scipy.optimize.minimize` "
                    f"(L-BFGS-B, с fallback на Nelder-Mead) так, чтобы "
                    f"смоделированный ВВП/работника максимально совпал "
                    f"с реальным рядом World Bank "
                    f"(`{ModelCalibrator.DEFAULT_INDICATOR}`) "
                    f"для страны **{_COUNTRIES[country][0]}**."
                )

                calibrate_clicked = st.button(
                    "🎯 Автокалибровка параметров (Scipy Optimize)",
                    key="btn_calibrate",
                    use_container_width=True,
                )

                if calibrate_clicked:
                    with st.spinner(
                        f"Калибруем α, r, γ, s под данные World Bank "
                        f"({_COUNTRIES[country][0]})…"
                    ):
                        try:
                            calibrator = ModelCalibrator(
                                engine=self.engine,
                                country=country,
                                k0=float(k0), A0=float(A0),
                                fixed_params=dict(DEFAULT_PARAMS),
                            )
                            calib_result = calibrator.fit(
                                x0=[alpha_val, r_val, gamma_val, s_val],
                                method="L-BFGS-B",
                            )
                            st.session_state["calib_result"] = calib_result
                        except ValueError as exc:
                            st.session_state["calib_result"] = None
                            st.error(f"Калибровка невозможна: {exc}")
                        except Exception as exc:  # noqa: BLE001 — показываем пользователю
                            st.session_state["calib_result"] = None
                            st.error(f"Ошибка калибровки: {exc}")

                calib_result: CalibrationResult | None = st.session_state.get("calib_result")

                if calib_result is not None and calib_result.country == country:
                    r2_txt = (
                        f"{calib_result.r2:.4f}"
                        if math.isfinite(calib_result.r2) else "н/д"
                    )
                    status_icon = "✅" if calib_result.success else "⚠️"
                    st.markdown(f"**{status_icon} Результат калибровки**")
                    st.caption(
                        f"R² = {r2_txt} · RMSE = {calib_result.rmse:.4f} · "
                        f"метод: {calib_result.method} · "
                        f"точек ряда: {calib_result.n_points}"
                    )

                    cc1, cc2 = st.columns(2)
                    cc1.metric("α (калибр.)", f"{calib_result.alpha:.4f}")
                    cc1.metric("γ (калибр.)", f"{calib_result.gamma:.4f}")
                    cc2.metric("r (калибр.)", f"{calib_result.r:.5f}")
                    cc2.metric("s (калибр.)", f"{calib_result.s:.4f}")

                    if not calib_result.success:
                        st.caption(
                            f"⚠ Оптимизатор не сообщил о полной сходимости: "
                            f"{calib_result.message}"
                        )

                    if st.button(
                        "✅ Применить значения к модели",
                        key="btn_apply_calibration",
                        use_container_width=True,
                        type="primary",
                    ):
                        st.session_state["_apply_calibration_pending"] = True
                        st.rerun()

                elif calib_result is not None and calib_result.country != country:
                    st.caption(
                        f"ℹ️ Есть сохранённая калибровка для "
                        f"{calib_result.country}, но сейчас выбрана другая "
                        f"страна ({country}). Нажмите кнопку выше, чтобы "
                        f"откалибровать заново для {country}."
                    )

            st.divider()

            # ----------------------------------------------------------
            # Блок 2: Реформы (модификаторы)
            # ----------------------------------------------------------
            st.subheader("📋 Реформы")

            savings_delta = st.slider(
                "💰 Стимул сбережений (Δs)",
                min_value=-0.10,
                max_value=+0.20,
                value=0.0,
                step=0.005,
                key="sb_savings",
                help=(
                    "Аддитивная надбавка к норме сбережений s. "
                    "+0.05 → s увеличивается на 5 п.п."
                ),
            )
            if savings_delta != 0.0:
                st.caption(f"s = базовое + {savings_delta:+.3f}")

            rd_scale = st.slider(
                "🔬 Инвестиции в R&D (×r)",
                min_value=0.5,
                max_value=3.0,
                value=1.0,
                step=0.05,
                key="sb_rd",
                help=(
                    "Мультипликатор темпа роста TFP. "
                    "×2.0 → скорость технологического прогресса удваивается."
                ),
            )
            if rd_scale != 1.0:
                st.caption(f"r = базовое × {rd_scale:.2f}")

            st.divider()

            # ----------------------------------------------------------
            # Блок 3: Шок (radio)
            # ----------------------------------------------------------
            st.subheader("💥 Шок")

            shock_keys   = list(SHOCK_CATALOG.keys())
            shock_labels = [SHOCK_CATALOG[k].label for k in shock_keys]

            shock_label = st.radio(
                "Тип шока",
                options=shock_labels,
                index=0,
                key="sb_shock_type",
            )
            shock_key = shock_keys[shock_labels.index(shock_label)]

            shock_start, shock_dur = 5.0, 4.0
            if shock_key != "no_shock":
                tmpl = SHOCK_CATALOG[shock_key]
                st.caption(tmpl.description)
                shock_start = st.slider(
                    "Начало шока (год)", 0.0, _T_END - 1.0, 5.0, 0.5, key="sb_shock_start",
                )
                shock_dur = st.slider(
                    "Длительность (лет)", 0.5, 15.0, 4.0, 0.5, key="sb_shock_dur",
                )
                st.caption(
                    f"Интервал: t = [{shock_start:.1f}, {shock_start + shock_dur:.1f}]"
                )

            st.markdown("**🌊 Форма шока (AdvancedShockManager)**")
            adv_profile_label = st.selectbox(
                "Временной профиль шока",
                options=[
                    "Ступенчатый (Базовый)",
                    "Плавный / Логистический (Реформа)",
                    "Циклический (Гармонический колебательный)",
                ],
                index=0,
                key="sb_adv_profile",
                help=(
                    "«Ступенчатый (Базовый)» полностью сохраняет прежнее "
                    "поведение (мгновенный переход в рецессию согласно "
                    "выбору выше). Остальные профили добавляют "
                    "непрерывную во времени модуляцию параметра s или δ "
                    "поверх базового сценария, не затрагивая солвер "
                    "scipy.integrate.solve_ivp."
                ),
            )
            adv_param      = "s"
            adv_start      = 5.0
            adv_delta      = -0.05
            adv_k_speed    = 1.0
            adv_amplitude  = 0.10
            adv_period     = 8.0

            if adv_profile_label != "Ступенчатый (Базовый)":
                adv_param_label = st.selectbox(
                    "Модулируемый параметр",
                    options=["s — норма сбережений", "δ (delta) — амортизация капитала"],
                    index=0,
                    key="sb_adv_param",
                )
                adv_param = "s" if adv_param_label.startswith("s") else "delta"

                adv_start = st.slider(
                    "Начало действия (год)", 0.0, _T_END - 1.0, 5.0, 0.5,
                    key="sb_adv_start",
                )

                if adv_profile_label.startswith("Плавный"):
                    adv_delta = st.slider(
                        "Δ — итоговое изменение параметра",
                        min_value=-0.30, max_value=0.30,
                        value=-0.05, step=0.01,
                        key="sb_adv_delta",
                        help="Значение параметра при t→+∞ = текущее значение + Δ.",
                    )
                    adv_k_speed = st.slider(
                        "k_speed — скорость перехода",
                        min_value=0.1, max_value=5.0,
                        value=1.0, step=0.1,
                        key="sb_adv_kspeed",
                        help="Больше k_speed → круче/быстрее сигмоидальный переход.",
                    )
                    st.caption(
                        f"{adv_param}(t) = {adv_param}₀ + ({adv_delta:+.3f}) / "
                        f"(1 + e^(-{adv_k_speed:.2f}·(t-{adv_start:.1f})))"
                    )
                else:  # Циклический
                    adv_amplitude = st.slider(
                        "Амплитуда A_amp (доля от базового значения)",
                        min_value=0.0, max_value=0.5,
                        value=0.10, step=0.01,
                        key="sb_adv_amp",
                    )
                    adv_period = st.slider(
                        "Период T_cycle (лет)",
                        min_value=1.0, max_value=20.0,
                        value=8.0, step=0.5,
                        key="sb_adv_period",
                        help="Длина одной волны бизнес-цикла (Жюгляра/Кондратьева).",
                    )
                    st.caption(
                        f"{adv_param}(t) = {adv_param}₀ · (1 + {adv_amplitude:.2f} · "
                        f"sin(2π/{adv_period:.1f}·(t-{adv_start:.1f}))), t ≥ {adv_start:.1f}"
                    )

            adv_profile = {
                "Ступенчатый (Базовый)": "step",
                "Плавный / Логистический (Реформа)": "logistic",
                "Циклический (Гармонический колебательный)": "cyclical",
            }[adv_profile_label]

            st.markdown("**🎲 Стохастический шум**")
            mc_enabled = st.checkbox(
                "Включить стохастический шум (Монте-Карло)",
                value=False,
                key="sb_mc_enabled",
                help=(
                    "Добавляет к динамике капитала мультипликативный "
                    "винеровский шум (метод Эйлера–Маруямы) и запускает "
                    "N независимых траекторий, не затрагивая основной "
                    "детерминированный солвер (scipy.integrate.solve_ivp)."
                ),
            )
            mc_sigma = 0.05
            mc_n_runs = 100
            if mc_enabled:
                mc_sigma = st.slider(
                    "Волатильность σ",
                    min_value=0.0, max_value=0.2,
                    value=0.05, step=0.005,
                    key="sb_mc_sigma",
                    help=(
                        "σ в k_{t+Δt} = k_t + drift·Δt + σ·k_t·√Δt·ξ_t, "
                        "ξ_t ~ N(0,1). σ=0 → все траектории совпадают "
                        "с детерминированной моделью."
                    ),
                )
                mc_n_runs = st.slider(
                    "Число траекторий N",
                    min_value=20, max_value=300,
                    value=100, step=10,
                    key="sb_mc_n_runs",
                )
                st.caption(
                    f"Будет запущено {mc_n_runs} прогонов методом "
                    f"Эйлера–Маруямы с σ={mc_sigma:.3f}."
                )

            st.divider()

            # ----------------------------------------------------------
            # Кнопка запуска
            # ----------------------------------------------------------
            run_clicked = st.button(
                "▶ Запустить симуляцию",
                type="primary",
                use_container_width=True,
            )

            # Краткая сводка активного сценария
            has_scenario = (savings_delta != 0.0 or rd_scale != 1.0 or shock_key != "no_shock")
            if has_scenario:
                st.info("Сценарий задан. Нажмите ▶ для запуска.")
            else:
                st.caption("Сценарий не задан — будет показан только baseline.")

        return {
            "country":       country,
            "k0":            float(k0),
            "A0":            float(A0),
            "alpha":         float(alpha_val),
            "r_base":        float(r_val),
            "gamma":         float(gamma_val),
            "s_base":        float(s_val),
            "savings_delta": float(savings_delta),
            "rd_scale":      float(rd_scale),
            "shock_key":     shock_key,
            "shock_start":   float(shock_start),
            "shock_dur":     float(shock_dur),
            "adv_profile":   adv_profile,
            "adv_param":     adv_param,
            "adv_start":     float(adv_start),
            "adv_delta":     float(adv_delta),
            "adv_k_speed":   float(adv_k_speed),
            "adv_amplitude": float(adv_amplitude),
            "adv_period":    float(adv_period),
            "mc_enabled":    bool(mc_enabled),
            "mc_sigma":      float(mc_sigma),
            "mc_n_runs":     int(mc_n_runs),
            "run_clicked":   run_clicked,
        }

    # ==================================================================
    # Запуск симуляций
    # ==================================================================

    def _run_simulations(self, ui: dict[str, Any]) -> None:
        """
        Запускает baseline- и сценарную симуляции и сохраняет в session_state.

        Baseline:   ScenarioManager без модификаторов и шоков — только
                    страновые данные DataProvider.
        Сценарий:   ScenarioManager с savings_delta, rd_scale, shock из sidebar.

        Вся математика делегирована в EconomicEngine.
        Параметры собираются через ScenarioManager.build_params().

        Args:
            ui: Словарь настроек из ``_render_sidebar()``.
        """
        # Реагируем на кнопку ИЛИ на смену страны
        country_changed = (
            st.session_state.get("_last_country") != ui["country"]
        )
        if not ui["run_clicked"] and not country_changed:
            return

        st.session_state["_last_country"] = ui["country"]

        # --- Загружаем данные страны ---
        with st.spinner(f"Загружаем данные {_COUNTRIES[ui['country']][0]}…"):
            try:
                country_data = DataProvider.fetch_country_data(ui["country"])
            except Exception as exc:
                st.error(f"Ошибка DataProvider: {exc}")
                return

        if country_data.get("source") == "global_fallback":
            st.warning(
                "API недоступен и страна не найдена. "
                "Использованы глобальные умолчания."
            )
        elif country_data.get("errors"):
            st.warning(
                f"Частичные данные ({country_data['source']}): "
                + "; ".join(country_data["errors"][:2])
            )

        st.caption(
            f"📡 Данные {ui['country']} ({country_data.get('source', '?')}): "
            f"n (рост населения) = {country_data['n']:.5f} — "
            f"единственная величина в модели, зависящая от сети/API."
        )

        # Структурные параметры (alpha, r, gamma, s) берутся из sidebar —
        # либо заданы вручную, либо получены через ModelCalibrator.fit()
        # и применены кнопкой «Применить значения к модели».
        # Темп роста населения n — по-прежнему из World Bank / DataProvider.
        base_overrides = {
            "s":     ui["s_base"],
            "n":     country_data["n"],
            "alpha": ui["alpha"],
            "r":     ui["r_base"],
            "gamma": ui["gamma"],
        }

        # ── Baseline ────────────────────────────────────────────────────
        sm_base = ScenarioManager(base_params={**DEFAULT_PARAMS, **base_overrides})
        params_base = sm_base.build_params(current_t=0.0)

        # --- Валидация params до solve ---
        param_warns = validate_params(params_base)
        if param_warns:
            st.warning("⚠ Предупреждения baseline params:\n" + "\n".join(param_warns))

        self.engine.t_span = (0.0, _T_END)

        try:
            with st.spinner("Интегрируем baseline (RK45)…"):
                df_base = self.engine.solve(
                    k0=ui["k0"], A0=ui["A0"],
                    params=params_base, shocks=[],
                )
                df_base = self.engine.compute_gdp(df_base, params_base)
        except (ValueError, RuntimeError) as exc:
            st.error(f"Ошибка симуляции baseline: {exc}")
            return

        # --- Валидация DataFrame ---
        df_issues = validate_dataframe(df_base, label="baseline")
        if df_issues:
            st.error("Проблемы в baseline DataFrame:\n" + "\n".join(df_issues))
            return

        # ── Сценарий ────────────────────────────────────────────────────
        sm_scen = ScenarioManager(base_params={**DEFAULT_PARAMS, **base_overrides})

        if ui["savings_delta"] != 0.0:
            sm_scen.set_modifier(
                "savings_stimulus", param="s",
                mode="add", value=ui["savings_delta"],
                label=f"Δs={ui['savings_delta']:+.3f}",
            )
        if ui["rd_scale"] != 1.0:
            sm_scen.set_modifier(
                "rd_investment", param="r",
                mode="scale", value=ui["rd_scale"],
                label=f"×r={ui['rd_scale']:.2f}",
            )
        if ui["shock_key"] != "no_shock":
            sm_scen.set_shock(
                ui["shock_key"],
                start_t=ui["shock_start"],
                duration=ui["shock_dur"],
            )

        params_scen = sm_scen.build_params(current_t=0.0)
        shocks_scen = sm_scen.get_shocks()

        # --- AdvancedShockManager: гладкий (логистический) или циклический
        #     шок поверх сценария. При adv_profile="step" (базовое поведение)
        #     dynamic_params_fn is None — солвер работает без изменений.
        adv_shock_mgr = AdvancedShockManager()
        if ui.get("adv_profile") == "logistic":
            adv_shock_mgr.set_logistic_shock(
                param=ui["adv_param"], start_t=ui["adv_start"],
                delta=ui["adv_delta"], k_speed=ui["adv_k_speed"],
            )
        elif ui.get("adv_profile") == "cyclical":
            adv_shock_mgr.set_cyclical_shock(
                param=ui["adv_param"], start_t=ui["adv_start"],
                amplitude=ui["adv_amplitude"], period=ui["adv_period"],
            )
        dynamic_params_fn = adv_shock_mgr.as_dynamic_params_fn()

        # --- Валидация params сценария ---
        scen_warns = validate_params(params_scen)
        if scen_warns:
            st.warning("⚠ Предупреждения scenario params:\n" + "\n".join(scen_warns))

        try:
            with st.spinner("Интегрируем сценарий (RK45)…"):
                df_scen = self.engine.solve(
                    k0=ui["k0"], A0=ui["A0"],
                    params=params_scen, shocks=shocks_scen,
                    dynamic_params=dynamic_params_fn,
                )
                df_scen = self.engine.compute_gdp(df_scen, params_scen)
        except (ValueError, RuntimeError) as exc:
            st.error(f"Ошибка симуляции сценария: {exc}")
            return

        # --- Валидация DataFrame сценария ---
        scen_issues = validate_dataframe(df_scen, label="scenario")
        if scen_issues:
            st.error("Проблемы в scenario DataFrame:\n" + "\n".join(scen_issues))
            return

        st.session_state["adv_shock_config"] = adv_shock_mgr.get_config()

        # ── Монте-Карло (стохастический шум, опционально) ─────────────────
        # Не влияет на детерминированный солвер выше — полностью отдельный
        # прогон через StochasticEngine (Euler–Maruyama), поверх параметров
        # и шоков активного сценария.
        mc_result: MonteCarloResult | None = None
        if ui.get("mc_enabled"):
            try:
                with st.spinner(
                    f"Монте-Карло: {ui['mc_n_runs']} прогонов "
                    f"(σ={ui['mc_sigma']:.3f}, Эйлер–Маруяма)…"
                ):
                    stochastic_engine = StochasticEngine(self.engine)
                    mc_result = stochastic_engine.run_monte_carlo(
                        k0=ui["k0"], A0=ui["A0"],
                        params=params_scen, shocks=shocks_scen,
                        sigma=ui["mc_sigma"], n_runs=ui["mc_n_runs"],
                        seed=42,
                    )
            except ValueError as exc:
                st.warning(f"Монте-Карло пропущено: {exc}")
                mc_result = None

        # Сохраняем всё в session_state
        st.session_state["df_baseline"]    = df_base
        st.session_state["df_scenario"]    = df_scen
        st.session_state["params_base"]    = params_base
        st.session_state["params_scen"]    = params_scen
        st.session_state["shocks_scen"]    = shocks_scen
        st.session_state["country_data"]   = country_data
        st.session_state["ui_snapshot"]    = dict(ui)  # для отображения
        st.session_state["mc_result"]      = mc_result

    # ==================================================================
    # Метрики
    # ==================================================================

    def _render_metrics(self) -> None:
        """
        Отрисовывает строку st.metric с показателями при t = _T_METRIC.

        Метрики:
        - Capital k при t=50  (baseline и delta vs сценарий)
        - GDP y    при t=50   (baseline и delta vs сценарий)
        - A (TFP)  при t=50
        - Фаза роста в t=50

        Delta показывается только если сценарий отличается от baseline.
        """
        df_b = st.session_state["df_baseline"]
        df_s = st.session_state["df_scenario"]
        p_b  = st.session_state["params_base"]
        p_s  = st.session_state["params_scen"]
        sh_s = st.session_state["shocks_scen"]

        # Снимки состояния через engine.get_state (читает из готового df)
        snap_b = self.engine.get_state(_T_METRIC, df_b, p_b, [])
        snap_s = self.engine.get_state(_T_METRIC, df_s, p_s, sh_s)

        # ВВП при t=50: колонка y (если есть) или gdp_per_worker из get_state
        def _gdp_at(df: pd.DataFrame) -> float:
            idx = (df["t"] - _T_METRIC).abs().idxmin()
            return float(df["y"].iloc[idx]) if "y" in df.columns else float(df["k"].iloc[idx])

        gdp_b = _gdp_at(df_b)
        gdp_s = _gdp_at(df_s)

        k_b  = snap_b["k"]
        k_s  = snap_s["k"]

        # Дельты (None если нет сценария)
        has_scenario = (df_b is not df_s)
        dk   = (k_s   - k_b)   if has_scenario else None
        dgdp = (gdp_s - gdp_b) if has_scenario else None
        dA   = (snap_s["A"] - snap_b["A"]) if has_scenario else None

        def _fmt_delta(v: float | None) -> str | None:
            if v is None:
                return None
            sign = "+" if v >= 0 else ""
            return f"{sign}{v:.3f}"

        st.subheader(f"📊 Показатели при t = {_T_METRIC:.0f} лет")
        c1, c2, c3, c4, c5 = st.columns(5)

        c1.metric(
            label="Capital k (t=50)",
            value=f"{k_b:.3f}",
            delta=_fmt_delta(dk),
            help="Капиталовооружённость на одного работника в момент t=50.",
        )
        c2.metric(
            label="GDP/worker y (t=50)",
            value=f"{gdp_b:.3f}",
            delta=_fmt_delta(dgdp),
            help="ВВП на работника (функция Кобба–Дугласа) в момент t=50.",
        )
        c3.metric(
            label="TFP A (t=50)",
            value=f"{snap_b['A']:.3f}",
            delta=_fmt_delta(dA),
            help="Уровень технологий (Total Factor Productivity) в момент t=50.",
        )
        c4.metric(
            label="Фаза (baseline)",
            value=snap_b["phase"].upper(),
            help="Фаза роста базового сценария в момент t=50.",
        )
        c5.metric(
            label="Фаза (сценарий)",
            value=snap_s["phase"].upper(),
            delta=(
                "≠ baseline" if snap_s["phase"] != snap_b["phase"] else "= baseline"
            ),
            help="Фаза роста сценария в момент t=50.",
        )

        # --- Страновые данные ---
        cd = st.session_state.get("country_data", {})
        if cd:
            src_icon = {"api": "🟢", "api_partial": "🟡",
                        "country_fallback": "🟠", "global_fallback": "🔴"}.get(
                cd.get("source", ""), "⚪"
            )
            st.caption(
                f"{src_icon} Данные: s={cd.get('s', 0):.4f} · "
                f"n={cd.get('n', 0):.5f} · "
                f"GDP={cd.get('gdp', 0):.3e} USD · "
                f"источник: {cd.get('source', '—')}"
            )

    # ==================================================================
    # Графики
    # ==================================================================

    def _render_charts(self) -> None:
        """
        Отрисовывает графики во вкладках:
        - «📈 Временные ряды» → build_timeseries_plot (baseline vs scenario)
        - «🌀 Фазовый портрет» → build_phase_plot (фазовый портрет сценария)
        - «🎲 Монте-Карло» → add_monte_carlo_bands (если включён стохастический
          шум в sidebar) — N траекторий + 95% доверительный интервал.
        - «🧮 Квазистатика» → QualitativeAnalyzer: стационарная точка k*,
          матрица Якоби, собственные значения λ1, λ2 и вердикт об
          устойчивости по Ляпунову.
        - «🎯 Калибровка» → ModelCalibrator: запуск калибровки по данным
          World Bank и график «реальность vs модель».

        Использует только PlotBuilder / QualitativeAnalyzer / ModelCalibrator
        — никакой математики внутри самого метода отрисовки.
        """
        df_b   = st.session_state["df_baseline"]
        df_s   = st.session_state["df_scenario"]
        ui     = st.session_state.get("ui_snapshot", {})
        sh_s   = st.session_state["shocks_scen"]
        mc     = st.session_state.get("mc_result")

        tab_ts, tab_phase, tab_mc, tab_stability, tab_calib = st.tabs(
            ["📈 Временные ряды", "🌀 Фазовый портрет", "🎲 Монте-Карло",
             "🧮 Квазистатика", "🎯 Калибровка"]
        )

        # ------------------------------------------------------------------
        with tab_ts:
            shock_intervals = None
            if ui.get("shock_key", "no_shock") != "no_shock" and sh_s:
                shock_intervals = [
                    (ui["shock_start"], ui["shock_start"] + ui["shock_dur"])
                ]

            fig_ts = self.plotter.build_timeseries_plot(
                df_base=df_b,
                df_reform=df_s,
                y_col="y",
                shock_intervals=shock_intervals,
                title="ВВП на работника y(t) — Baseline vs Сценарий",
            )
            st.plotly_chart(fig_ts, use_container_width=True)

            adv_cfg: AdvancedShockConfig | None = st.session_state.get("adv_shock_config")
            if adv_cfg is not None:
                st.caption(f"🌊 Активен продвинутый шок сценария: {adv_cfg.label}")

            # Дополнительно: сравнение k и A
            with st.expander("Сравнение k(t) и A(t)"):
                col_k, col_A = st.columns(2)
                with col_k:
                    fig_k = self.plotter.build_timeseries_plot(
                        df_b, df_s, y_col="k",
                        shock_intervals=shock_intervals,
                        title="Капиталовооружённость k(t)",
                    )
                    st.plotly_chart(fig_k, use_container_width=True)
                with col_A:
                    fig_A = self.plotter.build_timeseries_plot(
                        df_b, df_s, y_col="A",
                        shock_intervals=shock_intervals,
                        title="Технологии A(t)",
                    )
                    st.plotly_chart(fig_A, use_container_width=True)

        # ------------------------------------------------------------------
        with tab_phase:
            col_ph1, col_ph2 = st.columns(2)

            with col_ph1:
                st.caption("**Baseline** — без реформ и шоков")
                fig_ph_b = self.plotter.build_phase_plot(
                    df_b, title="Фазовый портрет — Baseline"
                )
                st.plotly_chart(fig_ph_b, use_container_width=True)

            with col_ph2:
                st.caption("**Сценарий** — с реформами / шоком")
                fig_ph_s = self.plotter.build_phase_plot(
                    df_s, title="Фазовый портрет — Сценарий"
                )
                st.plotly_chart(fig_ph_s, use_container_width=True)

            st.caption(
                "⭐ Равновесие k* — точка, где |dk/dt| минимален. "
                "◆ Аттрактор — конечная точка траектории. "
                "Стрелка показывает направление движения системы."
            )

        # ------------------------------------------------------------------
        with tab_mc:
            if mc is None:
                st.info(
                    "Стохастический шум выключен. Включите чекбокс "
                    "«Включить стохастический шум (Монте-Карло)» в разделе "
                    "«💥 Шок» боковой панели и нажмите «▶ Запустить симуляцию», "
                    "чтобы увидеть N траекторий и 95% доверительный интервал."
                )
            else:
                st.caption(
                    f"Метод Эйлера–Маруямы поверх параметров активного "
                    f"сценария: N={mc.n_runs} траекторий, σ={mc.sigma:.3f}, "
                    f"шаг Δt={mc.dt:.2f} года. Основной детерминированный "
                    f"солвер (RK45 / scipy.integrate.solve_ivp) не менялся — "
                    f"это независимый стохастический прогон."
                )

                mc_var_label = st.radio(
                    "Переменная",
                    options=["ВВП на работника y(t)", "Капиталовооружённость k(t)"],
                    index=0,
                    horizontal=True,
                    key="sb_mc_variable",
                )
                mc_var = "y" if "ВВП" in mc_var_label else "k"

                fig_mc = self.plotter.add_monte_carlo_bands(mc, variable=mc_var)
                st.plotly_chart(fig_mc, use_container_width=True)

                q = mc.quantiles[mc_var]
                cc1, cc2, cc3 = st.columns(3)
                cc1.metric("Медиана (t конечное)", f"{q['p50'][-1]:.3f}")
                cc2.metric("2.5-й перцентиль (t кон.)", f"{q['p2_5'][-1]:.3f}")
                cc3.metric("97.5-й перцентиль (t кон.)", f"{q['p97_5'][-1]:.3f}")

                spread_pct = (
                    (q["p97_5"][-1] - q["p2_5"][-1]) / max(q["p50"][-1], 1e-9) * 100.0
                )
                st.caption(
                    f"Ширина 95%-го интервала на конечном шаге ≈ "
                    f"{spread_pct:.1f}% от медианы — мера накопленной "
                    f"неопределённости при заданном σ."
                )

        # ------------------------------------------------------------------
        with tab_stability:
            self._render_stability_tab(df_b, df_s)

        # ------------------------------------------------------------------
        with tab_calib:
            self._render_calibration_tab()

    # ==================================================================
    # Вкладка «🎯 Калибровка»
    # ==================================================================

    def _render_calibration_tab(self) -> None:
        """
        UI вкладки калибровки (ModelCalibrator).

        Дублирует запуск калибровки из sidebar (тот же
        ``st.session_state["calib_result"]``), но дополнительно строит
        график «реальность vs модель» через
        ``PlotBuilder.build_calibration_fit_plot`` — визуальную проверку
        качества подгонки, которой нет в компактном виде sidebar.
        """
        country = st.session_state.get("ui_snapshot", {}).get("country", "KAZ")

        st.caption(
            "Автокалибровка α, r, γ, s под реальный ряд ВВП на душу "
            "населения (World Bank) методом `scipy.optimize.minimize`. "
            "Управление — в боковой панели, слева, в разделе "
            "«🎯 Автокалибровка параметров»; здесь — визуализация результата."
        )

        result: CalibrationResult | None = st.session_state.get("calib_result")

        if result is None:
            st.info(
                "Калибровка (in-sample) ещё не запускалась. Откройте в "
                "боковой панели раздел «🎯 Автокалибровка параметров "
                "(Scipy Optimize)», нажмите кнопку запуска — результат "
                "появится здесь и в сайдбаре одновременно. Ниже, "
                "независимо от этого, доступен блок «📉 Бэктест»."
            )
        else:
            cc1, cc2, cc3, cc4 = st.columns(4)
            r2_txt = f"{result.r2:.4f}" if math.isfinite(result.r2) else "н/д"
            cc1.metric("R²", r2_txt)
            cc2.metric("RMSE", f"{result.rmse:.4f}")
            cc3.metric("Метод", result.method)
            cc4.metric("Точек ряда", str(result.n_points))

            fig_calib = self.plotter.build_calibration_fit_plot(result)
            st.plotly_chart(fig_calib, use_container_width=True)

            st.markdown("**Откалиброванные параметры:**")
            pc1, pc2, pc3, pc4 = st.columns(4)
            pc1.metric("α", f"{result.alpha:.4f}")
            pc2.metric("r", f"{result.r:.5f}")
            pc3.metric("γ", f"{result.gamma:.4f}")
            pc4.metric("s", f"{result.s:.4f}")

            if result.country != country:
                st.caption(
                    f"ℹ️ Показана калибровка для {result.country}; в сайдбаре "
                    f"сейчас выбрана другая страна ({country}). Запустите "
                    f"калибровку заново, чтобы обновить график."
                )

        st.divider()

        # ------------------------------------------------------------------
        # Бэктест: честная out-of-sample проверка прогноза
        # ------------------------------------------------------------------
        st.markdown("### 📉 Бэктест (прогноз на N лет вперёд)")
        st.caption(
            "На всякий случай — проверка, что модель не просто **описывает** "
            "прошлое, а способна **предсказывать**. Калибровка происходит "
            "ТОЛЬКО по данным до отсечки; последние N лет модель не видит "
            "при подборе α, r, γ, s — они используются только для сравнения "
            "с прогнозом (реальный out-of-sample тест, а не подгонка задним числом)."
        )

        ui_snap = st.session_state.get("ui_snapshot", {})
        k0_bt = ui_snap.get("k0", 2.0)
        A0_bt = ui_snap.get("A0", 1.0)

        col_bt1, col_bt2 = st.columns([2, 1])
        with col_bt1:
            holdout_years = st.slider(
                "Отложенный период для прогноза (лет)",
                min_value=2, max_value=10, value=5, step=1,
                key="sb_backtest_holdout",
                help=(
                    "Сколько последних лет ряда World Bank отложить под "
                    "прогноз. Модель калибруется на оставшихся годах, "
                    "затем интегрируется вперёд через весь горизонт."
                ),
            )
        with col_bt2:
            st.write("")  # вертикальный отступ для выравнивания кнопки
            backtest_clicked = st.button(
                "📉 Запустить бэктест", key="btn_backtest",
                type="primary", use_container_width=True,
            )

        if backtest_clicked:
            with st.spinner(
                f"Калибруем на данных до отсечки и прогнозируем "
                f"{holdout_years} лет вперёд…"
            ):
                try:
                    bt_calibrator = ModelCalibrator(
                        engine=self.engine, country=country,
                        k0=float(k0_bt), A0=float(A0_bt),
                        fixed_params=dict(DEFAULT_PARAMS),
                    )
                    bt_result = bt_calibrator.backtest(
                        holdout_years=holdout_years, method="L-BFGS-B",
                    )
                    st.session_state["backtest_result"] = bt_result
                except ValueError as exc:
                    st.session_state["backtest_result"] = None
                    st.error(f"Бэктест невозможен: {exc}")
                except Exception as exc:  # noqa: BLE001
                    st.session_state["backtest_result"] = None
                    st.error(f"Ошибка бэктеста: {exc}")

        bt_result: BacktestResult | None = st.session_state.get("backtest_result")

        if bt_result is None:
            st.info(
                "Нажмите «📉 Запустить бэктест», чтобы увидеть, насколько "
                "хорошо модель, откалиброванная по прошлым данным, "
                "предсказывает недавние годы, которых не видела."
            )
            return

        if bt_result.country != country:
            st.caption(
                f"ℹ️ Показан бэктест для {bt_result.country}; в сайдбаре "
                f"выбрана другая страна ({country}). Запустите бэктест "
                f"заново для {country}."
            )

        bc1, bc2, bc3, bc4 = st.columns(4)
        r2_train_txt = f"{bt_result.r2_train:.4f}" if math.isfinite(bt_result.r2_train) else "н/д"
        bc1.metric("R² (обучение)", r2_train_txt)
        bc2.metric("RMSE (обучение)", f"{bt_result.rmse_train:.4f}")
        bc3.metric("RMSE (прогноз)", f"{bt_result.rmse_test:.4f}")
        mape_txt = f"{bt_result.mape_test:.1f}%" if math.isfinite(bt_result.mape_test) else "н/д"
        bc4.metric("MAPE (прогноз)", mape_txt)

        fig_bt = self.plotter.build_backtest_plot(bt_result)
        st.plotly_chart(fig_bt, use_container_width=True)

        st.caption(
            f"Обучение: {bt_result.train_years[0]}–{bt_result.train_years[-1]} "
            f"({len(bt_result.train_years)} лет) · Прогноз: "
            f"{bt_result.test_years[0]}–{bt_result.test_years[-1]} "
            f"({len(bt_result.test_years)} лет, модель их НЕ видела при калибровке) · "
            f"параметры: α={bt_result.alpha:.4f}, r={bt_result.r:.5f}, "
            f"γ={bt_result.gamma:.4f}, s={bt_result.s:.4f}"
        )

        if math.isfinite(bt_result.mape_test):
            if bt_result.mape_test < 15.0:
                st.success(
                    f"✅ MAPE прогноза {mape_txt} — модель разумно "
                    f"экстраполирует за пределы обучающей выборки."
                )
            elif bt_result.mape_test < 35.0:
                st.warning(
                    f"⚠️ MAPE прогноза {mape_txt} — заметное расхождение с "
                    f"реальностью; возможно, структура экономики изменилась "
                    f"за отложенный период сильнее, чем модель может учесть."
                )
            else:
                st.error(
                    f"❌ MAPE прогноза {mape_txt} — модель плохо "
                    f"экстраполирует; вероятный признак переобучения "
                    f"(хорошо описывает прошлое, но не предсказывает будущее) "
                    f"или структурного слома в данных."
                )

    # ==================================================================
    # Вкладка «🧮 Квазистатика»
    # ==================================================================

    def _render_stability_tab(self, df_b: pd.DataFrame, df_s: pd.DataFrame) -> None:
        """
        UI вкладки качественного анализа устойчивости (QualitativeAnalyzer).

        Даёт пользователю выбрать:
        - какой сценарий анализировать (Baseline / Сценарий);
        - момент времени t, из которого берётся «замороженное» A* (и фаза
          по умолчанию — из фактической фазы траектории в этот момент);
        - фазу для линеаризации (Авто из симуляции, либо принудительно
          extensive / intensive / recession);
        - метод поиска k* (аналитическая формула, всегда дублируется
          кросс-проверкой через ``scipy.optimize.root_scalar``).

        Результат — StabilityResult — сохраняется в
        ``st.session_state["stability_result"]`` и отображается ниже.
        """
        params_base = st.session_state.get("params_base") or DEFAULT_PARAMS
        params_scen = st.session_state.get("params_scen") or DEFAULT_PARAMS

        col_cfg1, col_cfg2 = st.columns(2)

        with col_cfg1:
            df_choice_label = st.radio(
                "Какую траекторию анализировать",
                options=["Baseline", "Сценарий (с реформами/шоком)"],
                index=0,
                key="sb_stab_df_choice",
                horizontal=True,
            )
            is_baseline = df_choice_label == "Baseline"
            df_active = df_b if is_baseline else df_s
            params_active = params_base if is_baseline else params_scen

            t_min, t_max = float(df_active["t"].min()), float(df_active["t"].max())
            t_default = min(max(_T_METRIC, t_min), t_max)
            t_pick = st.slider(
                "Момент времени t для «заморозки» A* (год)",
                min_value=t_min, max_value=t_max,
                value=float(t_default), step=0.5,
                key="sb_stab_t",
                help=(
                    "A(t) — уровень TFP в этот момент фиксируется как "
                    "параметр (квазистатический анализ), после чего "
                    "ищется k*, при котором dk/dt=0."
                ),
            )

        idx = (df_active["t"] - t_pick).abs().idxmin()
        row = df_active.loc[idx]
        A_at_t = float(row["A"])
        phase_at_t = str(row["phase"])

        with col_cfg2:
            phase_choice = st.selectbox(
                "Фаза для линеаризации",
                options=[f"Авто (из симуляции: {phase_at_t})", "extensive", "intensive", "recession"],
                index=0,
                key="sb_stab_phase",
            )
            phase_resolved = phase_at_t if phase_choice.startswith("Авто") else phase_choice

            st.metric("A(t) — зафиксированный уровень TFP", f"{A_at_t:.5f}")
            st.caption(f"t = {float(row['t']):.2f} лет · фаза в симуляции: **{phase_at_t}**")

        analyze_clicked = st.button(
            "🔬 Выполнить анализ устойчивости",
            key="btn_stability_analyze",
            type="primary",
            use_container_width=True,
        )

        if analyze_clicked:
            try:
                analyzer = QualitativeAnalyzer(params=params_active)
                result = analyzer.analyze(A_star=A_at_t, phase=phase_resolved, k_star_method="analytic")
                st.session_state["stability_result"] = result
                st.session_state["stability_result_meta"] = {
                    "df_choice": df_choice_label, "t": float(row["t"]),
                }
            except (ValueError, RuntimeError) as exc:
                st.session_state["stability_result"] = None
                st.error(f"Анализ устойчивости невозможен: {exc}")

        result: StabilityResult | None = st.session_state.get("stability_result")
        if result is None:
            st.info(
                "Нажмите «🔬 Выполнить анализ устойчивости», чтобы найти "
                "стационарную точку k*, построить матрицу Якоби и получить "
                "вердикт об устойчивости по Ляпунову."
            )
            return

        meta = st.session_state.get("stability_result_meta", {})
        st.divider()
        st.markdown(
            f"#### Результат для «{meta.get('df_choice', '?')}», "
            f"t≈{meta.get('t', float('nan')):.2f}, фаза «{result.phase}»"
        )

        # --- k*, собственные значения ---
        cc1, cc2, cc3 = st.columns(3)
        cc1.metric("k* (стационарная точка)", f"{result.k_star:.5f}")
        lam1, lam2 = result.eigenvalues[0], result.eigenvalues[1]

        def _fmt_eig(lam: complex) -> str:
            re, im = lam.real, lam.imag
            if abs(im) < 1e-9:
                return f"{re:+.5f}"
            return f"{re:+.5f} {'+' if im >= 0 else '-'} {abs(im):.5f}i"

        cc2.metric("λ1 (собственное значение)", _fmt_eig(lam1))
        cc3.metric("λ2 (собственное значение)", _fmt_eig(lam2))

        if result.k_star_root_scalar is not None:
            st.caption(
                f"Кросс-проверка `scipy.optimize.root_scalar` (метод Брента): "
                f"k* ≈ {result.k_star_root_scalar:.6f} "
                f"(аналитическая формула: {result.k_star:.6f}, "
                f"Δ={abs(result.k_star - result.k_star_root_scalar):.2e})."
            )
        else:
            st.caption(
                "⚠ Численная кросс-проверка root_scalar не сошлась для "
                "этой конфигурации параметров — показана только "
                "аналитическая оценка k*."
            )

        # --- Матрица Якоби ---
        st.markdown("**Матрица Якоби $J$ в точке $(k^*, A^*)$:**")
        J = result.jacobian
        st.latex(
            r"J = \begin{pmatrix} "
            rf"{J[0,0]:.5f} & {J[0,1]:.5f} \\ "
            rf"{J[1,0]:.5f} & {J[1,1]:.5f}"
            r" \end{pmatrix}"
            rf"\qquad \mathrm{{tr}}(J)={result.trace:.5f},\ \ "
            rf"\det(J)={result.determinant:.5f}"
        )

        # --- Вердикт ---
        verdict_text = f"**{result.stability_type}.** {result.verdict}"
        if "Седловая" in result.stability_type:
            st.warning(verdict_text)
        elif "Устойчив" in result.stability_type:
            st.success(verdict_text)
        elif "Неустойчив" in result.stability_type:
            st.error(verdict_text)
        else:
            st.info(verdict_text)

        with st.expander("Формулы, использованные в анализе"):
            _phase_formula = {
                "intensive": (
                    r"\dot{k} = s\,A^{\gamma}\,k^{\alpha} - (\delta+n)\,k "
                    r"\quad\Rightarrow\quad "
                    r"k^{*} = \left(\dfrac{s\,A^{\gamma}}{\delta+n}\right)^{\frac{1}{1-\alpha}}"
                ),
                "extensive": (
                    r"\dot{k} = s\,A\,k^{\alpha} - (\delta+n)\,k "
                    r"\quad\Rightarrow\quad "
                    r"k^{*} = \left(\dfrac{s\,A}{\delta+n}\right)^{\frac{1}{1-\alpha}}"
                ),
                "recession": (
                    r"\dot{k} = s_{rec}\,k^{\alpha} - (\delta_{rec}+n)\,k "
                    r"\quad\Rightarrow\quad "
                    r"k^{*} = \left(\dfrac{s_{rec}}{\delta_{rec}+n}\right)^{\frac{1}{1-\alpha}}"
                ),
            }
            st.latex(_phase_formula[result.phase])
            st.latex(
                r"J = \begin{pmatrix}"
                r"\dfrac{\partial \dot k}{\partial k} & \dfrac{\partial \dot k}{\partial A}\\[4pt]"
                r"\dfrac{\partial \dot A}{\partial k} & \dfrac{\partial \dot A}{\partial A}"
                r"\end{pmatrix},\qquad"
                r"\lambda_{1,2}:\ \det(J-\lambda I)=0"
            )
            st.caption(
                "Классификация: Re(λᵢ)<0 для обоих → устойчивый узел/фокус "
                "(аттрактор); Re(λᵢ)>0 для обоих → неустойчивый источник; "
                "разные знаки → седловая точка; Re(λ)≈0 → вырожденный случай."
            )

    # ==================================================================
    # Expander с сырыми данными
    # ==================================================================

    def _render_data_expander(self) -> None:
        """
        Expander «Сырые данные»: таблицы baseline и сценария.
        """
        with st.expander("🗂 Сырые данные симуляции"):
            df_b = st.session_state["df_baseline"]
            df_s = st.session_state["df_scenario"]

            tab_raw_b, tab_raw_s = st.tabs(["Baseline", "Сценарий"])
            with tab_raw_b:
                st.dataframe(df_b.round(5), use_container_width=True)
            with tab_raw_s:
                st.dataframe(df_s.round(5), use_container_width=True)

    # ==================================================================
    # Инициализация session_state
    # ==================================================================

    @staticmethod
    def _init_session_state() -> None:
        """
        Инициализирует все ключи session_state при первом запуске.

        Строгий список — не позволяет случайно обращаться к незаданным ключам.
        """
        defaults: dict[str, Any] = {
            "df_baseline":   None,
            "df_scenario":   None,
            "params_base":   None,
            "params_scen":   None,
            "shocks_scen":   [],
            "country_data":  {},
            "ui_snapshot":   {},
            "_last_country": None,
            "mc_result":     None,
            "calib_result":  None,
            "_apply_calibration_pending": False,
            "stability_result":      None,
            "stability_result_meta": {},
            "adv_shock_config": None,
            "backtest_result": None,
        }
        for key, val in defaults.items():
            if key not in st.session_state:
                st.session_state[key] = val


# ---------------------------------------------------------------------------
# Валидация параметров и DataFrame — изолированные утилиты
# ---------------------------------------------------------------------------

def validate_params(params: dict[str, float]) -> list[str]:
    """
    Проверяет словарь params перед передачей в EconomicEngine.

    Не вызывает solve — только проверяет диапазоны значений.
    Возвращает список предупреждений (пустой = всё OK).

    Args:
        params: Словарь параметров (из ScenarioManager.build_params()).

    Returns:
        Список строк с предупреждениями о подозрительных значениях.
    """
    warnings: list[str] = []

    checks: list[tuple[str, float, float, str]] = [
        # (ключ, min, max, описание)
        ("alpha", 0.05, 0.95,   "Доля капитала α должна быть в (0.05, 0.95)"),
        ("delta", 0.001, 0.50,  "Амортизация δ должна быть в (0.001, 0.50)"),
        ("n",    -0.05, 0.10,   "Рост населения n должен быть в (-0.05, 0.10)"),
        ("s",     0.01, 0.90,   "Норма сбережений s должна быть в (0.01, 0.90)"),
        ("r",     0.00, 0.30,   "Рост TFP r должен быть в [0, 0.30)"),
        ("gamma", 0.01, 2.00,   "Эластичность γ должна быть в (0.01, 2.00)"),
        ("A_critical", 0.10, 20.0, "A_critical должен быть в (0.10, 20.0)"),
        ("s_recession_factor",     0.01, 1.00, "s_recession_factor должен быть в (0.01, 1.0)"),
        ("delta_recession_factor", 1.00, 10.0, "delta_recession_factor должен быть в (1.0, 10.0)"),
        ("A_decay_recession",      0.00, 0.50, "A_decay_recession должен быть в [0, 0.5)"),
    ]

    for key, lo, hi, msg in checks:
        val = params.get(key)
        if val is None:
            warnings.append(f"Отсутствует параметр: {key!r}")
        elif not (lo <= val <= hi):
            warnings.append(f"{msg}. Текущее значение: {key}={val:.6g}")

    return warnings


def validate_dataframe(df: "pd.DataFrame", label: str = "df") -> list[str]:
    """
    Проверяет DataFrame после solve() + compute_gdp() на NaN, Inf и отрицательные значения.

    Args:
        df:    DataFrame из EconomicEngine.
        label: Метка для сообщений (напр. "baseline", "scenario").

    Returns:
        Список строк с обнаруженными проблемами (пустой = всё OK).
    """
    import math
    issues: list[str] = []

    numeric_cols = [c for c in df.columns if c != "phase"]

    for col in numeric_cols:
        series = df[col]

        nan_count = int(series.isna().sum())
        if nan_count:
            issues.append(f"[{label}] {col}: {nan_count} NaN значений")

        inf_count = int((~series.isna() & series.apply(
            lambda x: not math.isfinite(x) if isinstance(x, float) else False
        )).sum())
        if inf_count:
            issues.append(f"[{label}] {col}: {inf_count} Inf значений")

        if col in ("k", "A", "Y", "y", "L", "K"):
            neg_count = int((series.dropna() < 0).sum())
            if neg_count:
                issues.append(f"[{label}] {col}: {neg_count} отрицательных значений")

    if df.empty:
        issues.append(f"[{label}] DataFrame пуст")

    return issues


# ---------------------------------------------------------------------------
# Главная функция — workflow интеграции
# ---------------------------------------------------------------------------

def main() -> None:
    """
    Точка входа Digital Twin Economics.

    Полный workflow::

        startup_validation()
            ↓
        AppController.run()
            ↓ внутри run():
            _render_sidebar()           → ui: dict
                ↓
            _run_simulations(ui)
                ├─ DataProvider.fetch_country_data(country)
                │       ↓ country_data: {s, n, gdp, source}
                ├─ ScenarioManager(base_params)
                │       ↓
                │   .set_modifier(savings_stimulus)
                │   .set_modifier(rd_investment)
                │   .set_shock(shock_key)
                │       ↓
                │   .build_params(t=0)  → params: dict
                │   .get_shocks()       → shocks: list[Shock]
                │       ↓
                ├─ validate_params(params)
                │       ↓
                ├─ EconomicEngine.solve(k0, A0, params, shocks)   → df_raw
                │   .compute_gdp(df_raw, params)                   → df
                │       ↓
                ├─ validate_dataframe(df)
                │       ↓
                └─ session_state ← df_baseline, df_scenario, ...
                        ↓
            _render_metrics()           st.metric × 5
                ↓
            _render_charts()
                ├─ tab "Time Series"  → PlotBuilder.build_timeseries_plot()
                └─ tab "Phase Space"  → PlotBuilder.build_phase_plot()

    Startup validation проверяет доступность всех зависимостей
    до первого взаимодействия с пользователем.
    """
    _startup_validation()
    controller = AppController()
    controller.run()


def _startup_validation() -> None:
    """
    Проверяет окружение при старте приложения.

    Выполняется один раз — до отрисовки UI.
    Проверяет:
    1. Импорты всех зависимостей (scipy, plotly, pandas, requests).
    2. Корректность DEFAULT_PARAMS (validate_params).
    3. Smoke-test EconomicEngine на минимальных данных (без UI).
    4. Наличие всех шоков в SHOCK_CATALOG.

    При критической ошибке — выбрасывает RuntimeError, который
    Streamlit поймает и отобразит пользователю.
    """
    errors:   list[str] = []
    warnings: list[str] = []

    # ── 1. Зависимости ──────────────────────────────────────────────
    _required_modules = {
        "scipy":    "scipy",
        "plotly":   "plotly",
        "pandas":   "pandas",
        "requests": "requests",
        "numpy":    "numpy",
    }
    for display_name, module_name in _required_modules.items():
        try:
            __import__(module_name)
        except ImportError:
            errors.append(f"Отсутствует зависимость: {display_name}. "
                          f"Установите: pip install {module_name}")

    if errors:
        raise RuntimeError(
            "Критические зависимости не установлены:\n" + "\n".join(errors)
        )

    # ── 2. Валидация DEFAULT_PARAMS ──────────────────────────────────
    param_warnings = validate_params(DEFAULT_PARAMS)
    warnings.extend(param_warnings)

    # ── 3. Smoke-test EconomicEngine ─────────────────────────────────
    try:
        _engine = EconomicEngine(t_span=(0.0, 2.0), dt=1.0)
        _df = _engine.solve(k0=2.0, A0=1.0, params=DEFAULT_PARAMS, shocks=[])
        _df = _engine.compute_gdp(_df, DEFAULT_PARAMS)
        df_issues = validate_dataframe(_df, label="smoke_test")
        if df_issues:
            warnings.extend(df_issues)
        # Проверяем отсутствие NaN
        import numpy as np
        numeric = _df.select_dtypes(include=[float, int])
        if np.isnan(numeric.values).any():
            errors.append("Smoke-test EconomicEngine: NaN в результате solve()")
    except Exception as exc:
        errors.append(f"Smoke-test EconomicEngine упал: {exc}")

    # ── 4. SHOCK_CATALOG ─────────────────────────────────────────────
    required_shocks = {"no_shock", "capital_crisis", "tech_stagnation"}
    missing_shocks  = required_shocks - set(SHOCK_CATALOG)
    if missing_shocks:
        errors.append(f"SHOCK_CATALOG: отсутствуют шоки: {missing_shocks}")

    for key, tmpl in SHOCK_CATALOG.items():
        if key == "no_shock":
            continue
        # Проверяем, что param_overrides ссылаются на известные ключи
        unknown = set(tmpl.param_overrides) - set(DEFAULT_PARAMS)
        if unknown:
            warnings.append(
                f"SHOCK_CATALOG[{key!r}].param_overrides: "
                f"неизвестные ключи {unknown}"
            )

    # ── 5. _COUNTRIES ────────────────────────────────────────────────
    for code, (label, k0, A0) in _COUNTRIES.items():
        if k0 <= 0:
            errors.append(f"_COUNTRIES[{code!r}]: k0={k0} <= 0")
        if A0 <= 0:
            errors.append(f"_COUNTRIES[{code!r}]: A0={A0} <= 0")
        if code not in DataProvider.COUNTRY_FALLBACK:
            warnings.append(
                f"_COUNTRIES[{code!r}]: нет fallback в DataProvider.COUNTRY_FALLBACK"
            )

    # ── Итог ─────────────────────────────────────────────────────────
    _log.info(
        "Startup validation: errors=%d warnings=%d",
        len(errors), len(warnings),
    )
    for w in warnings:
        _log.warning("Startup warning: %s", w)

    if errors:
        _log.error("Startup FAILED: %s", errors)
        raise RuntimeError(
            "Startup validation провалена:\n" + "\n".join(errors)
        )


# ---------------------------------------------------------------------------
# Диагностика — изолированный слой, не меняет UI архитектуру
# ---------------------------------------------------------------------------

import time as _time


# ---------------------------------------------------------------------------
# DiagnosticsResult — результат одного теста / диагностики
# ---------------------------------------------------------------------------

@dataclass
class DiagnosticsResult:
    """
    Результат одной диагностической проверки или тестового сценария.

    Attributes:
        name:        Название теста / проверки.
        passed:      True — тест прошёл, False — провалился.
        message:     Короткое итоговое сообщение.
        details:     Список строк с подробностями.
        runtime_ms:  Время выполнения в миллисекундах.
        payload:     Произвольные дополнительные данные (params, df-preview и т.д.).
    """
    name:       str
    passed:     bool
    message:    str
    details:    list[str] = field(default_factory=list)
    runtime_ms: float     = 0.0
    payload:    dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Predefined test scenarios — только данные, без Streamlit
# ---------------------------------------------------------------------------

PREDEFINED_SCENARIOS: dict[str, dict[str, Any]] = {
    "kaz_baseline": {
        "label":       "🇰🇿 Казахстан — Baseline",
        "description": "Нефтяная экономика, умеренные сбережения, нет реформ.",
        "country":     "KAZ",
        "k0":          2.0,
        "A0":          1.0,
        "modifiers":   {},          # name → {param, mode, value}
        "shock_key":   "no_shock",
        "shock_start": 0.0,
        "shock_dur":   0.0,
        "reforms":     [],          # list of {name, param, delta, apply_at_t}
        "t_end":       60.0,
    },
    "usa_innovation": {
        "label":       "🇺🇸 США — Инновационный рост",
        "description": "Высокий R&D мультипликатор, умеренный рост сбережений.",
        "country":     "USA",
        "k0":          8.0,
        "A0":          2.5,
        "modifiers": {
            "rd_investment":    {"param": "r", "mode": "scale", "value": 2.5},
            "savings_stimulus": {"param": "s", "mode": "add",   "value": 0.03},
        },
        "shock_key":   "no_shock",
        "shock_start": 0.0,
        "shock_dur":   0.0,
        "reforms": [
            {"name": "R&D закон", "param": "gamma", "delta": 0.1, "apply_at_t": 5.0},
        ],
        "t_end": 60.0,
    },
    "chn_capital_crisis": {
        "label":       "🇨🇳 Китай — Капитальный кризис",
        "description": "Кризис в t=15 длиной 6 лет, после — восстановление.",
        "country":     "CHN",
        "k0":          5.0,
        "A0":          1.8,
        "modifiers":   {},
        "shock_key":   "capital_crisis",
        "shock_start": 15.0,
        "shock_dur":   6.0,
        "reforms":     [],
        "t_end": 60.0,
    },
    "sgp_intensive": {
        "label":       "🇸🇬 Сингапур — Интенсивный рост",
        "description": "Рекордные сбережения, высокий R&D, низкий A_critical.",
        "country":     "SGP",
        "k0":          6.5,
        "A0":          2.0,
        "modifiers": {
            "rd_investment":    {"param": "r",          "mode": "scale", "value": 2.0},
            "savings_stimulus": {"param": "s",          "mode": "add",   "value": 0.05},
            "low_a_crit":       {"param": "A_critical", "mode": "add",   "value": -0.8},
        },
        "shock_key":   "no_shock",
        "shock_start": 0.0,
        "shock_dur":   0.0,
        "reforms": [
            {"name": "Образование", "param": "gamma", "delta": 0.15, "apply_at_t": 3.0},
        ],
        "t_end": 60.0,
    },
}


# ---------------------------------------------------------------------------
# DiagnosticsEngine — вся логика проверок, без Streamlit
# ---------------------------------------------------------------------------

class DiagnosticsEngine:
    """
    Диагностический движок Digital Twin Economics.

    Ответственность:
    - Запуск предопределённых тестовых сценариев.
    - Проверка стабильности solve_ivp.
    - Проверка на NaN, отрицательные значения, взрывные решения.
    - Вычисление диагностических метрик (равновесие, runtime).
    - Анализ активного состояния из session_state.

    Не содержит Streamlit-кода. Возвращает DiagnosticsResult.
    """

    def __init__(self) -> None:
        self._engine = EconomicEngine(t_span=(0.0, 60.0), dt=1.0)

    # ==================================================================
    # Публичный интерфейс
    # ==================================================================

    def run_scenario(self, scenario_key: str) -> DiagnosticsResult:
        """
        Запускает предопределённый тестовый сценарий.

        Args:
            scenario_key: Ключ из ``PREDEFINED_SCENARIOS``.

        Returns:
            DiagnosticsResult с результатом и подробностями.
        """
        if scenario_key not in PREDEFINED_SCENARIOS:
            return DiagnosticsResult(
                name=scenario_key, passed=False,
                message=f"Неизвестный сценарий: {scenario_key!r}",
            )

        cfg = PREDEFINED_SCENARIOS[scenario_key]
        t0 = _time.perf_counter()
        details: list[str] = []
        payload: dict[str, Any] = {}

        try:
            # DataProvider
            country_data = DataProvider.fetch_country_data(cfg["country"])
            details.append(
                f"DataProvider: country={cfg['country']} "
                f"s={country_data['s']:.4f} n={country_data['n']:.5f} "
                f"source={country_data['source']}"
            )

            # ScenarioManager
            sm = ScenarioManager(base_params={
                **DEFAULT_PARAMS,
                "s": country_data["s"],
                "n": country_data["n"],
            })
            for mod_name, mod_cfg in cfg["modifiers"].items():
                sm.set_modifier(
                    mod_name,
                    param=mod_cfg["param"],
                    mode=mod_cfg["mode"],
                    value=mod_cfg["value"],
                )
            if cfg["shock_key"] != "no_shock":
                sm.set_shock(cfg["shock_key"], cfg["shock_start"], cfg["shock_dur"])
            for ref in cfg["reforms"]:
                sm.add_reform(**ref)

            params = sm.build_params(current_t=0.0)
            shocks = sm.get_shocks()

            # validate_params
            p_warns = validate_params(params)
            if p_warns:
                details.extend(p_warns)

            # Solve
            self._engine.t_span = (0.0, cfg["t_end"])
            df = self._engine.solve(k0=cfg["k0"], A0=cfg["A0"],
                                    params=params, shocks=shocks)
            df = self._engine.compute_gdp(df, params)

            # Stability checks
            stab = self._check_stability(df, label=scenario_key)
            details.extend(stab["details"])

            # Equilibrium
            eq = self._find_equilibrium(df)
            details.append(
                f"Равновесие k*≈{eq['k_star']:.4f} "
                f"(|dk/dt|_min={eq['dk_min']:.6f} at t={eq['t_star']:.1f})"
            )

            # Equations state at final point
            snap = self._engine.get_state(
                float(df["t"].iloc[-1]), df, params, shocks
            )
            details.append(
                f"Финальное состояние: k={snap['k']:.4f} A={snap['A']:.4f} "
                f"phase={snap['phase']} dk/dt={snap['dk_dt']:.6f} "
                f"dA/dt={snap['dA_dt']:.6f}"
            )

            runtime_ms = (_time.perf_counter() - t0) * 1000

            payload = {
                "params":    {k: round(v, 6) for k, v in params.items()},
                "df_head":   df.head(5).round(5).to_dict(orient="records"),
                "df_tail":   df.tail(5).round(5).to_dict(orient="records"),
                "n_points":  len(df),
                "phases":    sorted(df["phase"].unique().tolist()),
                "k_range":   [round(float(df["k"].min()), 4),
                               round(float(df["k"].max()), 4)],
                "A_range":   [round(float(df["A"].min()), 4),
                               round(float(df["A"].max()), 4)],
                "equilibrium": eq,
                "final_snap":  snap,
                "country_data": country_data,
            }

            passed = stab["passed"] and not p_warns
            message = (
                f"✅ Прошёл за {runtime_ms:.1f} мс | "
                f"k_fin={snap['k']:.3f} phase={snap['phase']}"
                if passed else
                f"⚠ Проблемы обнаружены | {len(stab['issues'])} issues"
            )

            return DiagnosticsResult(
                name=cfg["label"], passed=passed,
                message=message, details=details,
                runtime_ms=runtime_ms, payload=payload,
            )

        except Exception as exc:
            runtime_ms = (_time.perf_counter() - t0) * 1000
            return DiagnosticsResult(
                name=cfg.get("label", scenario_key),
                passed=False,
                message=f"❌ Исключение: {exc}",
                details=details + [f"EXCEPTION: {type(exc).__name__}: {exc}"],
                runtime_ms=runtime_ms,
            )

    def check_solver_stability(
        self,
        params: dict[str, float],
        k0: float,
        A0: float,
        t_end: float = 30.0,
        shocks: list[Shock] | None = None,
    ) -> DiagnosticsResult:
        """
        Проверяет стабильность solve_ivp на заданных params.

        Тесты:
        - NaN в результате
        - Отрицательные k или A
        - Взрывной рост (k > 1e6 или A > 1e6)
        - Успешное завершение интегратора

        Args:
            params: Словарь параметров модели.
            k0:     Начальная капиталовооружённость.
            A0:     Начальный TFP.
            t_end:  Горизонт симуляции.
            shocks: Список шоков.

        Returns:
            DiagnosticsResult.
        """
        t0 = _time.perf_counter()
        details: list[str] = []
        shocks = shocks or []

        p_warns = validate_params(params)
        details.extend(p_warns)

        try:
            self._engine.t_span = (0.0, t_end)
            df = self._engine.solve(k0=k0, A0=A0, params=params, shocks=shocks)
            df = self._engine.compute_gdp(df, params)
        except Exception as exc:
            return DiagnosticsResult(
                name="solver_stability", passed=False,
                message=f"❌ solve_ivp упал: {exc}",
                details=details + [str(exc)],
                runtime_ms=(_time.perf_counter() - t0) * 1000,
            )

        stab = self._check_stability(df, label="stability_check")
        details.extend(stab["details"])

        runtime_ms = (_time.perf_counter() - t0) * 1000
        return DiagnosticsResult(
            name="solver_stability",
            passed=stab["passed"],
            message=(
                f"✅ Стабильно ({runtime_ms:.1f} мс)"
                if stab["passed"] else
                f"⚠ {len(stab['issues'])} проблем"
            ),
            details=details,
            runtime_ms=runtime_ms,
            payload={"n_points": len(df), "phases": sorted(df["phase"].unique().tolist())},
        )

    def inspect_active_state(
        self,
        df_baseline: "pd.DataFrame | None",
        df_scenario: "pd.DataFrame | None",
        params_base: dict[str, float] | None,
        params_scen: dict[str, float] | None,
        shocks_scen: list[Shock] | None,
        t_inspect:   float = 50.0,
    ) -> DiagnosticsResult:
        """
        Анализирует активное состояние из session_state.

        Показывает:
        - Активные params baseline и сценария
        - get_state в момент t_inspect
        - Уравнения (dk/dt, dA/dt) в t_inspect
        - Информацию о шоках
        - DataFrame preview

        Args:
            df_baseline:  DataFrame baseline из session_state.
            df_scenario:  DataFrame сценария из session_state.
            params_base:  Params baseline.
            params_scen:  Params сценария.
            shocks_scen:  Список шоков сценария.
            t_inspect:    Момент времени для детального анализа.

        Returns:
            DiagnosticsResult.
        """
        t0 = _time.perf_counter()
        details: list[str] = []
        payload: dict[str, Any] = {}

        if df_baseline is None or df_scenario is None:
            return DiagnosticsResult(
                name="active_state", passed=False,
                message="Нет данных симуляции — запустите сначала.",
                details=["df_baseline или df_scenario = None"],
            )

        shocks_scen = shocks_scen or []

        # --- Params diff ---
        p_b = params_base or {}
        p_s = params_scen or {}
        changed = {k: (p_b.get(k), p_s.get(k))
                   for k in set(p_b) | set(p_s)
                   if abs((p_b.get(k, 0) or 0) - (p_s.get(k, 0) or 0)) > 1e-9}
        if changed:
            details.append(f"Изменённые параметры ({len(changed)}):")
            for k, (vb, vs) in sorted(changed.items()):
                details.append(f"  {k}: {vb:.6g} → {vs:.6g}  Δ={vs-vb:+.6g}")
        else:
            details.append("Params baseline == Scenario (нет модификаторов)")

        # --- Solver status ---
        details.append(
            f"Solver status: baseline={len(df_baseline)} pts, "
            f"scenario={len(df_scenario)} pts"
        )

        # --- Integration success (validate_dataframe) ---
        b_issues = validate_dataframe(df_baseline, "baseline")
        s_issues = validate_dataframe(df_scenario, "scenario")
        details.append(
            f"Integration: baseline_issues={len(b_issues)} "
            f"scenario_issues={len(s_issues)}"
        )
        details.extend(b_issues + s_issues)

        # --- Active state at t_inspect ---
        snap_b = self._engine.get_state(t_inspect, df_baseline, p_b, [])
        snap_s = self._engine.get_state(t_inspect, df_scenario, p_s, shocks_scen)
        details.append(
            f"\nСостояние при t={t_inspect}:"
        )
        details.append(
            f"  Baseline:  k={snap_b['k']:.4f}  A={snap_b['A']:.4f}  "
            f"phase={snap_b['phase']}  "
            f"dk/dt={snap_b['dk_dt']:.6f}  dA/dt={snap_b['dA_dt']:.6f}  "
            f"GDP/w={snap_b['gdp_per_worker']:.4f}"
        )
        details.append(
            f"  Scenario:  k={snap_s['k']:.4f}  A={snap_s['A']:.4f}  "
            f"phase={snap_s['phase']}  "
            f"dk/dt={snap_s['dk_dt']:.6f}  dA/dt={snap_s['dA_dt']:.6f}  "
            f"GDP/w={snap_s['gdp_per_worker']:.4f}"
        )
        details.append(
            f"  Δk={snap_s['k']-snap_b['k']:+.4f}  "
            f"ΔA={snap_s['A']-snap_b['A']:+.4f}  "
            f"ΔGDP/w={snap_s['gdp_per_worker']-snap_b['gdp_per_worker']:+.4f}"
        )

        # --- Equilibrium ---
        eq_b = self._find_equilibrium(df_baseline)
        eq_s = self._find_equilibrium(df_scenario)
        details.append(
            f"\nРавновесие: baseline k*≈{eq_b['k_star']:.4f} "
            f"| scenario k*≈{eq_s['k_star']:.4f}"
        )

        # --- Шоки ---
        if shocks_scen:
            for sh in shocks_scen:
                details.append(
                    f"Шок активен: {sh.label} "
                    f"t=[{sh.start_t}, {sh.start_t + sh.duration}]"
                )
        else:
            details.append("Шоки: нет")

        runtime_ms = (_time.perf_counter() - t0) * 1000
        payload = {
            "params_diff":  changed,
            "snap_baseline": snap_b,
            "snap_scenario": snap_s,
            "eq_baseline":   eq_b,
            "eq_scenario":   eq_s,
            "df_base_head":  df_baseline.head(5).round(5).to_dict(orient="records"),
            "df_scen_head":  df_scenario.head(5).round(5).to_dict(orient="records"),
        }

        passed = (not b_issues) and (not s_issues)
        return DiagnosticsResult(
            name="active_state",
            passed=passed,
            message="✅ Состояние OK" if passed else f"⚠ {len(b_issues+s_issues)} проблем",
            details=details,
            runtime_ms=runtime_ms,
            payload=payload,
        )

    # ==================================================================
    # Внутренние методы — только вычисления, без Streamlit
    # ==================================================================

    @staticmethod
    def _check_stability(
        df: "pd.DataFrame",
        label: str = "df",
    ) -> dict[str, Any]:
        """
        Проверяет DataFrame на NaN, отрицательные значения, взрывной рост.

        Returns:
            dict с ключами: passed, issues, details.
        """
        import math
        issues:  list[str] = []
        details: list[str] = []

        numeric_cols = [c for c in df.columns if c != "phase"]

        for col in numeric_cols:
            s = df[col].dropna()

            # NaN
            n_nan = int(df[col].isna().sum())
            if n_nan:
                issues.append(f"{col}: {n_nan} NaN")
                details.append(f"⚠ NaN [{label}] {col}: {n_nan}")

            # Inf
            n_inf = int(s.apply(lambda x: not math.isfinite(x)
                                 if isinstance(x, float) else False).sum())
            if n_inf:
                issues.append(f"{col}: {n_inf} Inf")
                details.append(f"⚠ Inf [{label}] {col}: {n_inf}")

            # Отрицательные (только для физических величин)
            if col in ("k", "A", "Y", "y", "L", "K"):
                n_neg = int((s < 0).sum())
                if n_neg:
                    issues.append(f"{col}: {n_neg} отрицательных")
                    details.append(f"⚠ Negative [{label}] {col}: {n_neg}")

            # Взрывной рост
            if col in ("k", "A", "Y", "y") and len(s) > 0:
                max_val = float(s.abs().max())
                if max_val > 1e6:
                    issues.append(f"{col}: взрывной рост max={max_val:.3e}")
                    details.append(f"⚠ Exploding [{label}] {col}: max={max_val:.3e}")

        if not issues:
            details.append(f"✅ Stability OK [{label}]: NaN=0 Inf=0 Neg=0 Explode=0")

        return {"passed": not issues, "issues": issues, "details": details}

    @staticmethod
    def _find_equilibrium(df: "pd.DataFrame") -> dict[str, float]:
        """
        Находит приближённую равновесную точку k* как минимум |dk/dt|.

        Не решает уравнение аналитически — только ищет минимум в траектории.
        """
        if "dk_dt" not in df.columns or df.empty:
            return {"k_star": 0.0, "dk_min": 0.0, "t_star": 0.0, "A_star": 0.0}

        idx    = df["dk_dt"].abs().idxmin()
        k_star = float(df.loc[idx, "k"])
        dk_min = float(df.loc[idx, "dk_dt"])
        t_star = float(df.loc[idx, "t"])
        A_star = float(df.loc[idx, "A"])

        return {
            "k_star": k_star,
            "dk_min": dk_min,
            "t_star": t_star,
            "A_star": A_star,
        }


# ---------------------------------------------------------------------------
# DiagnosticsUI — Streamlit-рендеринг диагностики
# Единственный класс, который знает о Streamlit в этом слое
# ---------------------------------------------------------------------------

class DiagnosticsUI:
    """
    Streamlit-слой диагностики.

    Оркестрирует DiagnosticsEngine и рисует результаты.
    Не содержит математики и вычислений — только вызовы DiagnosticsEngine
    и st.*-виджеты.

    Встраивается в AppController через метод render_diagnostics_tab().
    UI остаётся неизменным — диагностика добавляется как новая вкладка.
    """

    def __init__(self) -> None:
        self._diag = DiagnosticsEngine()

    # ==================================================================
    # Главный метод — вкладка «Диагностика»
    # ==================================================================

    def render(self) -> None:
        """
        Отрисовывает полную вкладку диагностики.

        Структура:
        1. Debug mode toggle + active state
        2. Тестовые сценарии (4 кнопки)
        3. Solver stability check
        4. Expander «Diagnostics» с деталями последнего результата
        """
        st.header("🔬 Диагностика и тестирование")

        # ── Debug toggle ────────────────────────────────────────────
        debug_mode = st.toggle(
            "🐛 Debug Mode",
            value=st.session_state.get("debug_mode", False),
            key="debug_toggle",
            help="Показывает расширенную информацию о состоянии системы.",
        )
        st.session_state["debug_mode"] = debug_mode

        st.divider()

        # ── Секция 1: Active state ───────────────────────────────────
        self._render_active_state(debug_mode)

        st.divider()

        # ── Секция 2: Тестовые сценарии ──────────────────────────────
        self._render_test_scenarios()

        st.divider()

        # ── Секция 3: Solver stability ───────────────────────────────
        self._render_solver_stability()

        st.divider()

        # ── Секция 4: Diagnostics expander ──────────────────────────
        self._render_diagnostics_expander()

    # ==================================================================
    # Секция 1 — Active state
    # ==================================================================

    def _render_active_state(self, debug_mode: bool) -> None:
        """Показывает активное состояние из session_state."""
        st.subheader("📡 Активное состояние")

        df_b = st.session_state.get("df_baseline")
        df_s = st.session_state.get("df_scenario")
        p_b  = st.session_state.get("params_base")
        p_s  = st.session_state.get("params_scen")
        sh_s = st.session_state.get("shocks_scen", [])

        if df_b is None:
            st.info("Симуляция ещё не запускалась. Перейдите в sidebar и нажмите ▶.")
            return

        # Быстрые метрики состояния
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Точек baseline", len(df_b))
        c2.metric("Точек scenario", len(df_s) if df_s is not None else 0)
        c3.metric(
            "Фазы baseline",
            ", ".join(sorted(df_b["phase"].unique())) if df_b is not None else "—",
        )
        c4.metric(
            "Активных шоков",
            len(sh_s),
            delta="⚡" if sh_s else None,
        )

        # Solver status
        b_ok = validate_dataframe(df_b, "baseline") == []
        s_ok = validate_dataframe(df_s, "scenario") == [] if df_s is not None else False
        col_b, col_s = st.columns(2)
        col_b.metric(
            "Integration baseline",
            "✅ OK" if b_ok else "❌ FAIL",
            help="Нет NaN / Inf / отрицательных значений",
        )
        col_s.metric(
            "Integration scenario",
            "✅ OK" if s_ok else "❌ FAIL",
        )

        # Кнопка детального анализа
        if st.button("🔍 Проанализировать активное состояние", key="btn_inspect"):
            t_inspect = float(st.session_state.get("diag_t_inspect", 50.0))
            with st.spinner("Анализируем…"):
                result = self._diag.inspect_active_state(
                    df_b, df_s, p_b, p_s, sh_s, t_inspect=t_inspect,
                )
            st.session_state["last_diag_result"] = result
            self._show_result(result)

        t_inspect = st.number_input(
            "Момент анализа t (лет)", value=50.0, min_value=0.0, step=5.0,
            key="diag_t_inspect",
        )

        # Debug mode — расширенные данные
        if debug_mode and df_b is not None:
            with st.expander("🐛 Debug: params diff"):
                if p_b and p_s:
                    rows = []
                    for k in sorted(set(p_b) | set(p_s)):
                        vb = p_b.get(k, None)
                        vs = p_s.get(k, None)
                        diff = (vs - vb) if (vb is not None and vs is not None) else None
                        rows.append({
                            "param":    k,
                            "baseline": round(vb, 6) if vb is not None else None,
                            "scenario": round(vs, 6) if vs is not None else None,
                            "Δ":        round(diff, 6) if diff is not None else None,
                        })
                    st.dataframe(pd.DataFrame(rows), use_container_width=True,
                                 hide_index=True)

            with st.expander("🐛 Debug: DataFrame preview (baseline)"):
                st.caption("Первые 10 строк baseline DataFrame:")
                st.dataframe(df_b.head(10).round(5), use_container_width=True)

            with st.expander("🐛 Debug: Equations state at t=0"):
                if p_b:
                    snap0 = self._diag._engine.get_state(
                        0.0, df_b, p_b, []
                    )
                    st.json(snap0)

    # ==================================================================
    # Секция 2 — Тестовые сценарии
    # ==================================================================

    def _render_test_scenarios(self) -> None:
        """Кнопки для запуска 4 предопределённых сценариев."""
        st.subheader("🧪 Тестовые сценарии")
        st.caption(
            "Каждый сценарий запускает полный workflow: "
            "DataProvider → ScenarioManager → EconomicEngine → валидация."
        )

        cols = st.columns(len(PREDEFINED_SCENARIOS))
        for col, (key, cfg) in zip(cols, PREDEFINED_SCENARIOS.items()):
            with col:
                st.caption(cfg["label"])
                st.caption(cfg["description"])
                if st.button("▶ Запустить", key=f"btn_scenario_{key}"):
                    with st.spinner(f"Запускаем {cfg['label']}…"):
                        t_start = _time.perf_counter()
                        result = self._diag.run_scenario(key)
                        result.runtime_ms = (_time.perf_counter() - t_start) * 1000
                    st.session_state["last_diag_result"] = result
                    if result.passed:
                        st.success(f"✅ {result.runtime_ms:.0f} мс")
                    else:
                        st.error("❌ Провал")

        # Показываем последний результат если есть
        if st.session_state.get("last_diag_result"):
            result = st.session_state["last_diag_result"]
            if result.name in [c["label"] for c in PREDEFINED_SCENARIOS.values()]:
                self._show_result(result)

    # ==================================================================
    # Секция 3 — Solver stability check
    # ==================================================================

    def _render_solver_stability(self) -> None:
        """Интерактивная проверка стабильности solve_ivp."""
        st.subheader("⚙️ Проверка стабильности решателя")

        col_a, col_b = st.columns(2)
        with col_a:
            stab_k0 = st.number_input("k₀", value=2.0, min_value=0.01, step=0.5,
                                       key="stab_k0")
            stab_A0 = st.number_input("A₀", value=1.0, min_value=0.01, step=0.1,
                                       key="stab_A0")
            stab_s  = st.slider("s (сбережения)", 0.01, 0.90, 0.22, 0.01, key="stab_s")
            stab_r  = st.slider("r (рост TFP)",   0.00, 0.20, 0.02, 0.005, key="stab_r")
        with col_b:
            stab_alpha = st.slider("α", 0.10, 0.70, 0.33, 0.01, key="stab_alpha")
            stab_delta = st.slider("δ", 0.01, 0.30, 0.05, 0.005, key="stab_delta")
            stab_n     = st.slider("n", -0.02, 0.05, 0.01, 0.001, key="stab_n")
            stab_t_end = st.slider("Горизонт (лет)", 10.0, 100.0, 30.0, 5.0,
                                    key="stab_t_end")

        if st.button("🔬 Проверить стабильность", key="btn_stability",
                     type="primary"):
            test_params = {
                **DEFAULT_PARAMS,
                "s": stab_s, "alpha": stab_alpha, "delta": stab_delta,
                "n": stab_n, "r": stab_r,
            }
            with st.spinner("Проверяем…"):
                result = self._diag.check_solver_stability(
                    params=test_params,
                    k0=stab_k0,
                    A0=stab_A0,
                    t_end=stab_t_end,
                )
            st.session_state["last_diag_result"] = result
            self._show_result(result)

    # ==================================================================
    # Секция 4 — Expander «Diagnostics»
    # ==================================================================

    def _render_diagnostics_expander(self) -> None:
        """
        Главный expander «Diagnostics» — показывает полный отчёт
        последнего диагностического результата.
        """
        with st.expander("🗂 Diagnostics — полный отчёт последнего теста",
                         expanded=False):
            result: DiagnosticsResult | None = st.session_state.get(
                "last_diag_result"
            )

            if result is None:
                st.info("Результатов нет — запустите любой тест выше.")
                return

            # Заголовок
            icon = "✅" if result.passed else "❌"
            st.markdown(f"**{icon} {result.name}**")
            st.caption(f"Runtime: {result.runtime_ms:.1f} мс")
            st.caption(result.message)

            # Details
            if result.details:
                st.markdown("**Детали:**")
                for line in result.details:
                    st.text(line)

            # Payload — params
            if result.payload.get("params"):
                with st.expander("Параметры модели (params)", expanded=False):
                    params_df = pd.DataFrame(
                        [{"параметр": k, "значение": v}
                         for k, v in result.payload["params"].items()
                         if k not in ("k_min", "A_min")]
                    )
                    st.dataframe(params_df, use_container_width=True, hide_index=True)

            # Payload — df preview
            if result.payload.get("df_head"):
                with st.expander("DataFrame preview (первые 5 строк)", expanded=False):
                    st.dataframe(
                        pd.DataFrame(result.payload["df_head"]),
                        use_container_width=True,
                    )
                with st.expander("DataFrame preview (последние 5 строк)",
                                  expanded=False):
                    st.dataframe(
                        pd.DataFrame(result.payload["df_tail"]),
                        use_container_width=True,
                    )

            # Payload — equilibrium
            if result.payload.get("equilibrium"):
                eq = result.payload["equilibrium"]
                st.markdown("**Равновесная точка k\\*:**")
                eq_cols = st.columns(4)
                eq_cols[0].metric("k*",   f"{eq['k_star']:.4f}")
                eq_cols[1].metric("A*",   f"{eq['A_star']:.4f}")
                eq_cols[2].metric("|dk/dt|", f"{abs(eq['dk_min']):.6f}")
                eq_cols[3].metric("t*",   f"{eq['t_star']:.1f}")

            # Payload — final equations state
            if result.payload.get("final_snap"):
                snap = result.payload["final_snap"]
                st.markdown("**Уравнения (финальное состояние):**")
                eq_state_cols = st.columns(3)
                eq_state_cols[0].metric("dk/dt",    f"{snap['dk_dt']:.6f}")
                eq_state_cols[1].metric("dA/dt",    f"{snap['dA_dt']:.6f}")
                eq_state_cols[2].metric("Фаза",     snap["phase"])

            # Payload — country data
            if result.payload.get("country_data"):
                cd = result.payload["country_data"]
                st.caption(
                    f"Данные страны: s={cd.get('s', 0):.4f}  "
                    f"n={cd.get('n', 0):.5f}  "
                    f"GDP={cd.get('gdp', 0):.3e}  "
                    f"source={cd.get('source', '—')}"
                )

            # k_range, A_range, phases
            if result.payload.get("k_range"):
                st.caption(
                    f"k ∈ [{result.payload['k_range'][0]}, "
                    f"{result.payload['k_range'][1]}]  |  "
                    f"A ∈ [{result.payload['A_range'][0]}, "
                    f"{result.payload['A_range'][1]}]  |  "
                    f"Фазы: {result.payload.get('phases', [])}"
                )

    # ==================================================================
    # Вспомогательный рендер результата
    # ==================================================================

    @staticmethod
    def _show_result(result: DiagnosticsResult) -> None:
        """Показывает краткое summary результата прямо под кнопкой."""
        if result.passed:
            st.success(f"✅ **{result.name}** — {result.message}")
        else:
            st.error(f"❌ **{result.name}** — {result.message}")

        # Первые 5 строк details
        if result.details:
            with st.expander("Подробности", expanded=False):
                for line in result.details[:20]:
                    st.text(line)
                if len(result.details) > 20:
                    st.caption(f"… и ещё {len(result.details) - 20} строк")


# ---------------------------------------------------------------------------
# Расширение AppController — добавляем вкладку диагностики
# ---------------------------------------------------------------------------

# Monkey-patch: добавляем вкладку «Диагностика» в AppController.run()
# без изменения существующей архитектуры
_original_run = AppController.run


def _patched_run(self: AppController) -> None:
    """
    Расширенный run() с вкладкой диагностики.

    Оригинальный AppController.run() устанавливает set_page_config и title,
    затем рисует sidebar и вызывает _run_simulations + _render_*.
    Мы оборачиваем его, чтобы добавить вкладку «🔬 Диагностика» без
    изменения существующих методов.
    """
    st.set_page_config(
        page_title="Digital Twin Economics",
        page_icon="📈",
        layout="wide",
    )
    AppController._init_session_state()

    # Инициализируем состояние диагностики
    if "last_diag_result" not in st.session_state:
        st.session_state["last_diag_result"] = None
    if "debug_mode" not in st.session_state:
        st.session_state["debug_mode"] = False

    # Sidebar (оригинальный)
    ui = self._render_sidebar()

    # Заголовок
    st.title("📈 Digital Twin Economics")
    st.caption(
        f"Модель Солоу · Страна: **{_COUNTRIES[ui['country']][0]}** · "
        f"Горизонт: {_T_END:.0f} лет · Метрики при t={_T_METRIC:.0f}"
    )
    st.divider()

    # Симуляции
    self._run_simulations(ui)

    # Основной контент с вкладкой диагностики
    if st.session_state["df_baseline"] is not None:
        self._render_metrics()
        self._render_charts()
        self._render_data_expander()

    st.divider()

    # Диагностика — отдельная секция внизу страницы
    with st.expander("🔬 Диагностика и тестирование", expanded=False):
        diag_ui = DiagnosticsUI()
        diag_ui.render()


AppController.run = _patched_run




# ---------------------------------------------------------------------------
# Точка входа
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    main()
