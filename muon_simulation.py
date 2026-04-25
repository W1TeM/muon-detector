import numpy as np
from scipy.stats import moyal
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import matplotlib.colors as mcolors
from matplotlib.animation import PillowWriter

class Particle:
    """
    Базовый класс физической частицы.
    Хранит кинематические параметры и предоставляет методы для работы с ними.
    """
    def __init__(self, mass: float, charge: float, kinetic_energy: float, 
                 theta: float, phi: float, position: np.ndarray):
        """
        :param mass: Масса покоя частицы в МэВ/c^2
        :param charge: Электрический заряд в единицах элементарного заряда (e)
        :param kinetic_energy: Кинетическая энергия в МэВ
        :param theta: Зенитный угол в радианах [0, pi/2] (угол падения)
        :param phi: Азимутальный угол в радианах[0, 2*pi]
        :param position: Вектор координат [x, y, z] в метрах
        """
        self.mass = mass
        self.charge = charge
        self.kinetic_energy = kinetic_energy
        self.theta = theta
        self.phi = phi
        self.position = np.array(position, dtype=float)
        
        # Индикатор активности частицы (остановилась ли она в веществе)
        self.is_active = True

    @property
    def total_energy(self) -> float:
        """Полная энергия частицы (E = K + mc^2) в МэВ."""
        return self.kinetic_energy + self.mass

    @property
    def momentum(self) -> float:
        """Импульс частицы в МэВ/c, вычисляемый из релятивистского инварианта: p*c = sqrt(E^2 - (mc^2)^2)."""
        return np.sqrt(self.total_energy**2 - self.mass**2)

    @property
    def direction(self) -> np.ndarray:
        """Единичный вектор направления движения частицы, исходя из углов theta и phi."""
        return np.array([
            np.sin(self.theta) * np.cos(self.phi),
            np.sin(self.theta) * np.sin(self.phi),
            -np.cos(self.theta)  # Отрицательный Z, так как мюоны летят сверху вниз
        ])


class Muon(Particle):
    """
    Класс мюона. Наследуется от Particle с предустановленной массой и зарядом.
    В физике КЛ на уровне моря отношение положительных мюонов к отрицательным ~1.25,
    но для потерь энергии по Бете-Блоху знак заряда роли не играет (входит как z^2).
    """
    MUON_MASS = 105.6583755  # МэВ/c^2

    def __init__(self, kinetic_energy: float, theta: float, phi: float, position: np.ndarray, charge: float = -1.0):
        super().__init__(mass=self.MUON_MASS, charge=charge, 
                         kinetic_energy=kinetic_energy, theta=theta, phi=phi, position=position)


class MaterialLayer:
    """
    Класс слоя вещества (например, бетонное перекрытие).
    Содержит физические параметры, необходимые для расчета ионизационных потерь.
    """
    def __init__(self, name: str, thickness: float, density: float, z_over_a: float, mean_excitation_energy: float):
        """
        :param name: Название материала (например, "Concrete")
        :param thickness: Толщина слоя в метрах
        :param density: Плотность в г/см^3 (удобно для Бете-Блоха)
        :param z_over_a: Отношение атомного номера к атомной массе (Z/A) в моль^-1
        :param mean_excitation_energy: Средний потенциал ионизации (I) в эВ
        """
        self.name = name
        self.thickness = thickness
        self.density = density
        self.z_over_a = z_over_a
        self.mean_excitation_energy = mean_excitation_energy  # Будет переведено в МэВ в транспортном движке


class CosmicMuonGenerator:
    """
    Генератор пула космических мюонов с правильными физическими распределениями.
    """
    def __init__(self, e_min: float = 1e3, e_max: float = 1e5):
        """
        :param e_min: Минимальная кинетическая энергия генерации в МэВ (по умолч. 1 ГэВ)
        :param e_max: Максимальная кинетическая энергия генерации в МэВ (по умолч. 100 ГэВ)
        """
        self.e_min = e_min
        self.e_max = e_max

    def _energy_pdf(self, E: float) -> float:
        """
        Функция плотности вероятности для энергии мюона.
        Используется модифицированная аппроксимация (power-law) для уровня моря: f(E) ~ (E + E0)^-2.7
        Здесь энергия E подставляется в МэВ. E0 ~ 3.6 ГэВ (3600 МэВ).
        """
        E0 = 3600.0  # МэВ
        return (E + E0)**(-2.7)

    def _generate_energies(self, n_particles: int) -> np.ndarray:
        """Генерация энергий методом выборки с отклонением (Rejection Sampling)."""
        energies = np.zeros(n_particles)
        count = 0
        
        # Максимум функции плотности достигается на минимальной энергии (спектр падающий)
        f_max = self._energy_pdf(self.e_min)
        
        while count < n_particles:
            # Генерируем равномерно распределенные кандидаты по энергии и по оси Y для PDF
            E_candidates = np.random.uniform(self.e_min, self.e_max, n_particles - count)
            y_candidates = np.random.uniform(0, f_max, n_particles - count)
            
            # Оставляем только те, которые попали под кривую PDF
            valid_mask = y_candidates < self._energy_pdf(E_candidates)
            valid_E = E_candidates[valid_mask]
            
            # Добавляем прошедшие проверку энергии в итоговый массив
            n_valid = len(valid_E)
            energies[count : count + n_valid] = valid_E
            count += n_valid
            
        return energies

    def _generate_angles(self, n_particles: int) -> tuple[np.ndarray, np.ndarray]:
        """
        Генерация углов:
        Азимутальный phi: равномерно [0, 2*pi]
        Зенитный theta: PDF ~ cos^2(theta)*sin(theta). CDF = 1 - cos^3(theta).
        Метод обратного преобразования: theta = arccos((1 - U)^(1/3)), где U ~ Uniform(0, 1)
        """
        u = np.random.uniform(0, 1, n_particles)
        theta = np.arccos(np.cbrt(1 - u))  # np.cbrt извлекает кубический корень
        
        phi = np.random.uniform(0, 2 * np.pi, n_particles)
        return theta, phi

    def generate(self, n_particles: int, start_z: float = 0.0) -> list[Muon]:
        """
        Главный метод генератора. Создает список объектов мюонов.
        :param n_particles: Количество мюонов для генерации
        :param start_z: Начальная высота (z-координата), с которой стартуют мюоны (м)
        """
        energies = self._generate_energies(n_particles)
        thetas, phis = self._generate_angles(n_particles)
        
        muons =[]
        for i in range(n_particles):
            # Для простоты считаем, что мюоны падают на площадку над детектором равномерно.
            # В более сложной модели можно задать XY-профиль (например квадрат 10х10 м над зданием).
            # Пока оставим x=0, y=0. Транспортный движок и аксептанс детектора учтут геометрию.
            pos = np.array([0.0, 0.0, start_z])
            muon = Muon(kinetic_energy=energies[i], theta=thetas[i], phi=phis[i], position=pos)
            muons.append(muon)
            
        return muons

class TransportEngine:
    """
    Транспортный движок для проведения частиц через слои вещества.
    Реализует шаговое интегрирование потерь энергии и обновление координат.
    """
    # Физические константы
    ME_C2 = 0.51099895  # Масса электрона в МэВ/c^2
    K_CONST = 0.307075  # Константа Бете-Блоха в МэВ*см^2/моль

    def __init__(self, step_size_m: float = 0.05):
        """
        :param step_size_m: Размер пространственного шага интегратора в метрах.
                            5 см (0.05 м) — хороший баланс между скоростью и точностью.
        """
        self.step_size_m = step_size_m

    def _calculate_dedx(self, particle: Particle, material: MaterialLayer) -> float:
        """
        Расчет удельных ионизационных потерь энергии по формуле Бете-Блоха.
        
        Обоснование: Используется классическая формула Бете. 
        Пренебрегаем многократным рассеянием (мюон - тяжелая частица, траектория прямая) 
        и поправкой на эффект плотности среды (для упрощения симуляции).
        
        :return: Потери энергии dE/dx в МэВ/м
        """
        # Кинематические факторы
        gamma = particle.total_energy / particle.mass
        beta = np.sqrt(1.0 - 1.0 / gamma**2)
        
        # Максимально возможная передача энергии электрону (T_max) в МэВ
        # Точная кинематическая формула:
        num = 2 * self.ME_C2 * beta**2 * gamma**2
        den = 1 + 2 * gamma * (self.ME_C2 / particle.mass) + (self.ME_C2 / particle.mass)**2
        t_max = num / den
        
        # Средний потенциал ионизации переводим из эВ в МэВ
        I_mev = material.mean_excitation_energy * 1e-6
        
        # Логарифмический член
        log_term = np.log((2 * self.ME_C2 * beta**2 * gamma**2 * t_max) / (I_mev**2))
        
        # Основная формула потерь (МэВ * см^2 / г)
        dedx_mass = self.K_CONST * (particle.charge**2) * material.z_over_a * (1.0 / beta**2) * (0.5 * log_term - beta**2)
        
        # Перевод в МэВ/см (умножение на плотность г/см^3)
        dedx_cm = dedx_mass * material.density
        
        # Перевод в МэВ/м
        dedx_m = dedx_cm * 100.0
        
        # Теоретически потери не могут быть отрицательными (если энергия слишком мала, формула ломается)
        return max(dedx_m, 0.0)

    def propagate(self, particle: Particle, material: MaterialLayer) -> None:
        """
        Проводит частицу через заданный слой материала.
        Учитывает угол падения: реальный путь в слое = толщина / cos(theta).
        """
        if not particle.is_active:
            return

        # Защита от деления на ноль или горизонтального полета (мюоны летят сверху вниз, theta < pi/2)
        if particle.theta >= np.pi / 2:
            particle.is_active = False
            return
            
        # Реальный путь частицы через слой (геометрический фактор)
        path_length = material.thickness / np.cos(particle.theta)
        distance_traveled = 0.0
        
        while distance_traveled < path_length and particle.is_active:
            # На последнем шаге идем не на полный step_size, а на остаток пути
            step = min(self.step_size_m, path_length - distance_traveled)
            
            # Вычисляем потери энергии на этом шаге
            dedx = self._calculate_dedx(particle, material)
            energy_loss = dedx * step
            
            # Применяем потери
            particle.kinetic_energy -= energy_loss
            distance_traveled += step
            
            # Обновляем координаты (сдвиг по вектору направления)
            particle.position += particle.direction * step
            
            # Проверяем, не застрял ли мюон в бетоне
            if particle.kinetic_energy <= 0:
                particle.kinetic_energy = 0.0
                particle.is_active = False
                break


class Detector:
    """
    Модель сцинтилляционного детектора.
    Определяет геометрический аксептанс и генерирует реалистичный отклик (энерговыделение)
    с учетом флуктуаций Ландау.
    """
    def __init__(self, size_x: float, size_y: float, thickness: float, z_position: float):
        """
        :param size_x: Ширина детектора (м)
        :param size_y: Длина детектора (м)
        :param thickness: Толщина пластикового сцинтиллятора (м)
        :param z_position: Высота установки детектора (м)
        """
        self.size_x = size_x
        self.size_y = size_y
        self.thickness = thickness
        self.z_position = z_position
        
        # Типичные параметры пластического сцинтиллятора (например, полистирол CH)
        self.density = 1.032  # г/см^3
        self.z_over_a = 0.54  # моль^-1
        self.I_eV = 64.7      # Средний потенциал ионизации в эВ
        
        # Физические константы
        self.ME_C2 = 0.51099895
        self.K_CONST = 0.307075

    def check_acceptance(self, particle: Particle) -> bool:
        """
        Проверяет, пересекает ли траектория частицы активную область детектора.
        Использует аналитическую проекцию луча на плоскость Z.
        """
        # Если частица остановилась в перекрытиях или летит вверх (отразилась/ошибка генерации)
        if not particle.is_active or particle.direction[2] >= 0:
            return False
            
        # Расстояние по Z от текущей позиции мюона до плоскости детектора
        dz = self.z_position - particle.position[2]
        
        # Если детектор выше частицы, значит мюон уже пролетел мимо
        if dz > 0:
            return False
            
        # Находим параметр t для параметрического уравнения луча: r(t) = r0 + v*t
        t = dz / particle.direction[2]
        
        # Точка пересечения плоскости детектора
        hit_x = particle.position[0] + particle.direction[0] * t
        hit_y = particle.position[1] + particle.direction[1] * t
        
        # Проверка попадания в прямоугольную апертуру
        if abs(hit_x) <= self.size_x / 2.0 and abs(hit_y) <= self.size_y / 2.0:
            # Перемещаем частицу ровно на поверхность детектора
            particle.position += particle.direction * t
            return True
            
        return False

    def get_signal(self, particle: Particle) -> float:
        """
        Вычисляет энерговыделение (в МэВ) в сцинтилляторе.
        Генерирует случайное значение из распределения Мойала (аппроксимация Ландау) 
        на основе кинематики частицы.
        """
        if not self.check_acceptance(particle):
            return 0.0

        # Вычисляем релятивистские факторы
        gamma = particle.total_energy / particle.mass
        beta = np.sqrt(1.0 - 1.0 / gamma**2)
        
        # Реальная длина пути в детекторе с учетом угла падения
        path_length_m = self.thickness / np.cos(particle.theta)
        
        # Массовая толщина: г/см^2 = (м * 100) * г/см^3
        x_mass = (path_length_m * 100.0) * self.density
        
        # Масштабный параметр распределения Ландау (xi) в МэВ
        xi = 0.5 * self.K_CONST * self.z_over_a * (x_mass / beta**2)
        
        # Перевод потенциала ионизации в МэВ
        I_mev = self.I_eV * 1e-6
        
        # Вычисление Наиболее Вероятной Потери Энергии (MPV)
        term1 = np.log((2 * self.ME_C2 * beta**2 * gamma**2) / I_mev)
        term2 = np.log(xi / I_mev)
        mpv = xi * (term1 + term2 + 0.2 - beta**2)
        
        # Генерируем энерговыделение из распределения Мойала
        deposited_energy = moyal.rvs(loc=mpv, scale=xi)
        
        # Физический лимит: частица не может отдать энергии больше, чем у нее есть
        if deposited_energy >= particle.kinetic_energy:
            deposited_energy = particle.kinetic_energy
            particle.kinetic_energy = 0.0
            particle.is_active = False
        else:
            particle.kinetic_energy -= deposited_energy
            
        return max(deposited_energy, 0.0)


class BackgroundGenerator:
    """
    Генератор естественного радиационного фона.
    """
    def __init__(self, rate_hz: float = 50.0, mean_energy_mev: float = 1.2):
        """
        :param rate_hz: Средняя частота фоновых событий (срабатываний детектора) в Герцах
        :param mean_energy_mev: Среднее энерговыделение фонового события (имитация комптоновского континуума)
        """
        self.rate_hz = rate_hz
        self.mean_energy_mev = mean_energy_mev

    def generate(self, time_window_seconds: float) -> np.ndarray:
        """
        Возвращает массив энерговыделений от фона за заданное время наблюдений.
        Количество событий случайно (распределение Пуассона).
        """
        expected_counts = self.rate_hz * time_window_seconds
        n_events = np.random.poisson(expected_counts)
        
        if n_events == 0:
            return np.array([])
            
        # Экспоненциальное распределение хорошо имитирует форму комптоновского континуума для низких энергий
        background_energies = np.random.exponential(scale=self.mean_energy_mev, size=n_events)
        
        return background_energies

import copy

class Simulation:
    """
    Класс-оркестратор (Менеджер симуляции).
    Управляет генерацией, транспортом, детектированием и сбором статистики.
    """
    def __init__(self, detector: Detector, engine: TransportEngine, 
                 muon_gen: CosmicMuonGenerator, bg_gen: BackgroundGenerator):
        self.detector = detector
        self.engine = engine
        self.muon_gen = muon_gen
        self.bg_gen = bg_gen
        
        # Интегральный поток мюонов на уровне моря ~ 100 частиц / (м^2 * с)
        self.muon_flux_m2_s = 100.0 
        # Площадь виртуальной генерации над зданием (10х10 метров)
        self.generation_area = 100.0 
        self.total_rate_hz = self.muon_flux_m2_s * self.generation_area

    def run_building_scan(self, n_primary_muons: int, n_layers: int = 5, custom_material: MaterialLayer = None) -> dict:
        """
        УЛЬТРА-ОПТИМИЗИРОВАННЫЙ (векторизованный) запуск симуляции.
        Время выполнения для 1 млн частиц снижено с 8 минут до ~1 секунды.
        """
        results = {}
        time_window = n_primary_muons / self.total_rate_hz
        
        if custom_material is None:
            custom_material = MaterialLayer(name="Concrete", thickness=0.3, density=2.4, 
                                            z_over_a=0.5, mean_excitation_energy=135.0)

        # Константы для быстрого расчета
        detector_z = self.detector.z_position
        start_z = 15.0
        
        for current_layer in range(n_layers + 1):
            # 1. МАССОВАЯ ГЕНЕРАЦИЯ (Извлекаем сырые numpy-массивы напрямую, минуя ООП)
            energies = self.muon_gen._generate_energies(n_primary_muons)
            thetas, phis = self.muon_gen._generate_angles(n_primary_muons)
            
            x_pos = np.random.uniform(-5.0, 5.0, n_primary_muons)
            y_pos = np.random.uniform(-5.0, 5.0, n_primary_muons)

            # 2. МАССОВЫЙ ТРАНСПОРТ (Аналитический расчет потерь в бетоне)
            if current_layer > 0:
                # Мюоны теряют примерно 2 МэВ на г/см^2. 
                # Рассчитываем среднюю потерю на метр: 2.0 * плотность(2.4) * 100 = 480 МэВ/м
                avg_loss_per_m = 2.0 * custom_material.density * 100.0
                
                # Реальный путь через все слои бетона с учетом угла падения
                total_thickness_m = custom_material.thickness * current_layer
                path_length_m = total_thickness_m / np.cos(thetas)
                
                # Вычитаем потери из начальной энергии
                energies -= (avg_loss_per_m * path_length_m)

            # 3. МАССОВАЯ ГЕОМЕТРИЯ (Кто попал в детектор?)
            alive_mask = energies > 0
            
            # Проекция траектории на плоскость детектора
            # t = dz / cos(theta), dz = 15.0 м
            t_flight = (start_z - detector_z) / np.cos(thetas)
            hit_x = x_pos + (np.sin(thetas) * np.cos(phis)) * t_flight
            hit_y = y_pos + (np.sin(thetas) * np.sin(phis)) * t_flight
            
            # Маска попадания в апертуру детектора
            half_x, half_y = self.detector.size_x / 2.0, self.detector.size_y / 2.0
            geom_mask = (np.abs(hit_x) <= half_x) & (np.abs(hit_y) <= half_y)
            
            # Оставляем только те частицы, которые выжили И попали в цель
            valid_hits = alive_mask & geom_mask
            hit_energies = energies[valid_hits]
            hit_thetas = thetas[valid_hits]

            # 4. ФЛУКТУАЦИИ ЛАНДАУ В ДЕТЕКТОРЕ
            if len(hit_energies) > 0:
                # Векторизованный расчет MPV (Наиболее вероятной потери)
                muon_mass = Muon.MUON_MASS
                gamma = (hit_energies + muon_mass) / muon_mass
                beta = np.sqrt(1.0 - 1.0 / gamma**2)
                
                path_det = self.detector.thickness / np.cos(hit_thetas)
                x_mass = (path_det * 100.0) * self.detector.density
                xi = 0.5 * self.detector.K_CONST * self.detector.z_over_a * (x_mass / beta**2)
                
                I_mev = self.detector.I_eV * 1e-6
                term1 = np.log((2 * self.detector.ME_C2 * beta**2 * gamma**2) / I_mev)
                term2 = np.log(xi / I_mev)
                mpv = xi * (term1 + term2 + 0.2 - beta**2)
                
                # Вызов moyal для всего массива сразу
                deposited = moyal.rvs(loc=mpv, scale=xi)
                # Ограничиваем сверху остатком кинетической энергии, снизу - нулем
                deposited = np.clip(deposited, 0.0, hit_energies)
                muon_signals = deposited
            else:
                muon_signals = np.array([])

            # 5. ФОН
            bg_signals = self.bg_gen.generate(time_window_seconds=time_window)

            results[current_layer] = {
                'muon_signals': muon_signals,
                'bg_signals': np.array(bg_signals),
                'time': time_window
            }

        return results
    
    def plot_results(self, results: dict, threshold_mev: float = 3.0):
        """
        Визуализирует собранные данные.
        1. Спектр энерговыделений (на крыше и в подвале).
        2. Затухание скорости счета (Rate) мюонов в зависимости от толщины бетона.
        """
        floors = sorted(results.keys())
        
        # Подготовка данных для графика затухания
        rates = []
        errors =[]
        
        for floor in floors:
            data = results[floor]
            # Суммируем все сигналы (мюоны + фон)
            all_signals = np.concatenate([data['muon_signals'], data['bg_signals']])
            
            # Применяем порог дискриминации (отсекаем гамма-фон)
            muons_detected = len(all_signals[all_signals > threshold_mev])
            
            # Расчет Rate и пуассоновской ошибки: err = sqrt(N)/t
            rate = muons_detected / data['time']
            error = np.sqrt(muons_detected) / data['time'] if muons_detected > 0 else 0
            
            rates.append(rate)
            errors.append(error)

        # Создаем фигуру с двумя графиками
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

        # --- ГРАФИК 1: Энергетические спектры (Крыша vs Подвал) ---
        roof_data = results[0]
        basement_data = results[max(floors)]
        
        roof_all = np.concatenate([roof_data['muon_signals'], roof_data['bg_signals']])
        basement_all = np.concatenate([basement_data['muon_signals'], basement_data['bg_signals']])

        # Строим гистограммы с логарифмической шкалой Y (чтобы видеть и огромный фон, и мюоны)
        # Крыша: рисуем только толстым синим контуром (step)
        ax1.hist(roof_all, bins=50, range=(0, 20), histtype='step', linewidth=2, 
                 color='blue', label='Крыша (0 слоев)')
        
        # Подвал: делаем заливку приятным оранжевым цветом
        ax1.hist(basement_all, bins=50, range=(0, 20), alpha=0.6, 
                 color='darkorange', label=f'Подвал ({max(floors)} слоев)')
        # Линия порога дискриминации
        ax1.axvline(x=threshold_mev, color='black', linestyle='--', label=f'Порог дискриминации ({threshold_mev} МэВ)')
        
        ax1.set_yscale('log')
        ax1.set_xlabel('Энерговыделение в сцинтилляторе, МэВ')
        ax1.set_ylabel('Количество событий')
        ax1.set_title('Спектр сигналов детектора (Распределение Ландау + Фон)')
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # --- ГРАФИК 2: Затухание потока мюонов ---
        thicknesses = [f * 0.3 for f in floors] # Ось X: толщина бетона в метрах
        
        ax2.errorbar(thicknesses, rates, yerr=errors, fmt='o-', color='darkgreen', 
                     capsize=5, capthick=2, markersize=8)
        
        ax2.set_xlabel('Толщина бетона над детектором, м')
        ax2.set_ylabel('Скорость счета мюонов (Rate), Гц')
        ax2.set_title('Ослабление мюонного потока (с учетом ошибок Пуассона)')
        ax2.grid(True, linestyle='--')

        plt.tight_layout()
        plt.show()

class AnimatedVisualizer:
    """
    Класс для создания живой видеосимуляции пролета частиц через здание.
    """
    def __init__(self, n_floors: int = 5, floor_thickness: float = 0.3):
        self.n_floors = n_floors
        self.floor_thickness = floor_thickness
        self.building_height = 15.0
        
        # Геометрия этажей (Z-координаты)
        self.floor_z_positions = np.linspace(2.0, self.building_height - 2.0, n_floors)
        
        # Инструменты симуляции
        self.muon_gen = CosmicMuonGenerator(e_min=100.0, e_max=20000.0) # до 20 ГэВ для наглядности
        self.engine = TransportEngine(step_size_m=0.2) # Шаг побольше для ускорения анимации
        self.concrete = MaterialLayer("Concrete", floor_thickness, 2.4, 0.5, 135.0)
        
        self.particles =[]

    def setup_plot(self):
        self.fig, self.ax = plt.subplots(figsize=(8, 10))
        self.ax.set_xlim(-6, 6)
        self.ax.set_ylim(-1, self.building_height + 1)
        self.ax.set_aspect('equal')
        self.ax.set_facecolor('#1e1e1e') # Темный фон для красоты
        self.fig.patch.set_facecolor('#1e1e1e')
        
        self.ax.set_title(f'Прохождение мюонов через {self.n_floors} этажей', color='white', fontsize=14)
        self.ax.set_xlabel('Ось X, метры', color='white')
        self.ax.set_ylabel('Высота Z, метры', color='white')
        self.ax.tick_params(colors='white')

        # Отрисовка перекрытий (бетон)
        for z in self.floor_z_positions:
            rect = plt.Rectangle((-5, z), 10, self.floor_thickness, 
                                 color='gray', alpha=0.5, zorder=1)
            self.ax.add_patch(rect)
            
        # Отрисовка детектора внизу
        det_rect = plt.Rectangle((-0.5, 0), 1.0, 0.15, color='cyan', zorder=2)
        self.ax.add_patch(det_rect)
        self.ax.text(1.0, 0, "Детектор", color='cyan', fontsize=12)

        # Настройка цветовой шкалы (от 0 до 10 ГэВ)
        # Используем логарифмическую шкалу цветов, чтобы видеть разницу на малых энергиях
        self.norm = mcolors.LogNorm(vmin=10, vmax=20000)
        self.cmap = plt.cm.jet # jet отлично показывает переход от синего (холодного) к красному (горячему)
        
        # Создаем Scatter (точки частиц)
        self.scatter = self.ax.scatter([], [], c=[], cmap=self.cmap, norm=self.norm, 
                                       s=30, zorder=3, edgecolors='white', linewidths=0.5)
        
        # Добавляем Colorbar
        cbar = self.fig.colorbar(self.scatter, ax=self.ax, fraction=0.046, pad=0.04)
        cbar.set_label('Кинетическая энергия (МэВ)', color='white')
        cbar.ax.yaxis.set_tick_params(color='white')
        plt.setp(plt.getp(cbar.ax.axes, 'yticklabels'), color='white')

    def init_animation(self):
        """Начальный кадр"""
        self.scatter.set_offsets(np.empty((0, 2)))
        return self.scatter,

    def update(self, frame):
        """Вычисляется для каждого кадра анимации"""
        # Каждые несколько кадров "вбрасываем" новую порцию мюонов с крыши
        if frame % 5 == 0 and frame < 150:
            new_muons = self.muon_gen.generate(15) # 15 частиц за раз
            for m in new_muons:
                m.position[0] = np.random.uniform(-5, 5)
                m.position[2] = self.building_height
            self.particles.extend(new_muons)

        active_positions =[]
        active_energies =[]

        for p in self.particles:
            if not p.is_active:
                # Если частица умерла (застряла), все равно показываем ее серым на последнем месте
                active_positions.append([p.position[0], p.position[2]])
                active_energies.append(1) # Минимальная энергия, будет синим/прозрачным
                continue

            # Проверяем, находится ли частица внутри какого-то слоя бетона
            in_concrete = False
            for floor_z in self.floor_z_positions:
                if floor_z <= p.position[2] <= floor_z + self.floor_thickness:
                    in_concrete = True
                    break
            
            # Двигаем частицу на 1 шаг
            step_size = self.engine.step_size_m
            
            if in_concrete:
                # Если в бетоне - теряет энергию
                dedx = self.engine._calculate_dedx(p, self.concrete)
                p.kinetic_energy -= dedx * step_size
                if p.kinetic_energy <= 0:
                    p.kinetic_energy = 0
                    p.is_active = False

            # Сдвиг координат
            p.position += p.direction * step_size
            
            # Если улетела ниже пола - отключаем
            if p.position[2] < -1.0:
                p.is_active = False

            active_positions.append([p.position[0], p.position[2]])
            active_energies.append(p.kinetic_energy)

        # Обновляем график
        if active_positions:
            self.scatter.set_offsets(np.array(active_positions))
            self.scatter.set_array(np.array(active_energies))

        return self.scatter,

    def run(self):
        self.setup_plot()
        # Создаем анимацию (200 кадров, 30 миллисекунд между кадрами)
        self.anim = FuncAnimation(self.fig, self.update, init_func=self.init_animation,
                                  frames=250, interval=30, blit=True)
        plt.show()
# ==========================================
# ТОЧКА ВХОДА (Итоговый запуск проекта)
# ==========================================
if __name__ == "__main__":
    print("==============================================")
    print("  СИМУЛЯТОР КОСМИЧЕСКИХ МЮОНОВ В ЗДАНИИ (v2.0) ")
    print("==============================================")
    print("Выберите режим работы:")
    print("1 - Начать сбор статистики и построить научные графики")
    print("2 - Запустить живую видеосимуляцию (Анимация)")
    print("3 - Изменить параметры (Количество этажей)")
    
    n_floors = 5  # По умолчанию
    
    while True:
        choice = input("\nВаш выбор (1, 2, 3 или q для выхода): ")
        
        if choice == 'q':
            break
            
        elif choice == '3':
            try:
                val = int(input("Введите количество этажей (от 1 до 10): "))
                if 1 <= val <= 10:
                    n_floors = val
                    print(f"Установлено: {n_floors} этажей.")
                else:
                    print("Ошибка: введите число от 1 до 10.")
            except:
                print("Некорректный ввод.")
                
        elif choice == '2':
            print("Запуск видеосимуляции... (Закройте окно графика, чтобы вернуться в меню)")
            animator = AnimatedVisualizer(n_floors=n_floors, floor_thickness=0.3)
            animator.run()
            
        elif choice == '1':
            n_muons = 100_000  # Снизил до 100к для скорости тестирования, для финальных графиков верни 1_000_000
            print(f"Запуск расчета Монте-Карло для {n_floors} этажей...")
            
            gen_muon = CosmicMuonGenerator(e_min=100.0, e_max=100000.0)
            gen_bg = BackgroundGenerator(rate_hz=500.0, mean_energy_mev=1.0)
            detector = Detector(size_x=1.0, size_y=1.0, thickness=0.05, z_position=0.0)
            engine = TransportEngine(step_size_m=0.05)
            sim = Simulation(detector, engine, gen_muon, gen_bg)
            
            results = sim.run_building_scan(n_primary_muons=n_muons, n_floors=n_floors)
            sim.plot_results(results, threshold_mev=3.0)