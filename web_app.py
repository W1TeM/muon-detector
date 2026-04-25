import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import PillowWriter, FuncAnimation
import matplotlib.colors as mcolors
from scipy.stats import moyal

# Импортируем Plotly для будущих графиков
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

# Импорты физического ядра
from muon_simulation import (
    CosmicMuonGenerator, TransportEngine, MaterialLayer, 
    Detector, BackgroundGenerator, Simulation
)

# Обязательно: конфигурация страницы (должна быть первой командой)
st.set_page_config(page_title="Мюонная Томография", layout="wide", initial_sidebar_state="expanded")

# ==========================================
# ФУНКЦИИ-ПОМОЩНИКИ (Анимация из предыдущей версии)
# ==========================================
def generate_animation_gif(n_layers, duration_sec, material, det_thick_mm, bg_intensity):
    fps = 15
    total_frames = int(duration_sec * fps)
    injection_frames = total_frames - int(1.5 * fps) 
    building_height = 15.0
    layer_z_positions = np.linspace(2.0, building_height - 2.0, n_layers)
    
    # Расчет порога на основе толщины (1 МэВ на 15 мм)
    threshold_mev = (det_thick_mm / 15.0) * 1.0
    
    muon_gen = CosmicMuonGenerator(e_min=100.0, e_max=20000.0)
    engine = TransportEngine(step_size_m=0.2)
    
    fig = plt.figure(figsize=(14, 8))
    fig.patch.set_facecolor('#1e1e1e')
    gs = fig.add_gridspec(1, 2, width_ratios=[1, 1.2])
    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 1])
    
    ax1.set_xlim(-6, 6)
    ax1.set_ylim(-1, building_height + 1)
    ax1.set_facecolor('#1e1e1e')
    ax1.set_title(f'Среда: {material.name} | Слоев: {n_layers}', color='white', fontsize=14)
    ax1.tick_params(colors='white')
    
    for z in layer_z_positions:
        layer_alpha = min(0.3 + material.density / 20.0, 0.9)
        ax1.add_patch(plt.Rectangle((-5, z), 10, material.thickness, color='gray', alpha=layer_alpha))
        
    ax1.add_patch(plt.Rectangle((-0.5, 0), 1.0, 0.15, color='cyan'))
    counter_text = ax1.text(-2.0, -0.8, "Поймано мюонов: 0", color='cyan', fontsize=14, fontweight='bold')
    
    norm = mcolors.LogNorm(vmin=10, vmax=20000)
    cmap = plt.cm.jet
    scatter = ax1.scatter([],[], c=[], cmap=cmap, norm=norm, s=40, edgecolors='white', linewidths=0.5)

    energies_muon, energies_bg = [],[]
    hist_bins = np.linspace(0, 20, 50)
    dt = 1.0 / fps
    # Фон теперь зависит от глобального ползунка
    bg_rate_hz = 50.0 * bg_intensity 

    particles = []
    processed_particles = set()
    hits = 0
    
    def init():
        scatter.set_offsets(np.empty((0, 2)))
        counter_text.set_text("Поймано мюонов: 0")
        return scatter, counter_text
        
    def update(frame):
        nonlocal hits
        if frame % 6 == 0 and frame < injection_frames:
            new_muons = muon_gen.generate(15)
            for m in new_muons:
                m.position[0] = np.random.uniform(-4.5, 4.5)
                m.position[2] = building_height
            particles.extend(new_muons)
            
        active_positions, active_energies = [],[]
        for p in particles:
            if p.is_active:
                in_material = any(z <= p.position[2] <= z + material.thickness for z in layer_z_positions)
                if in_material:
                    dedx = engine._calculate_dedx(p, material)
                    p.kinetic_energy -= dedx * engine.step_size_m
                    if p.kinetic_energy <= 0:
                        p.kinetic_energy = 0
                        p.is_active = False
                
                p.position += p.direction * engine.step_size_m
                
                if p not in processed_particles and p.position[2] <= 0.15:
                    processed_particles.add(p)
                    if -0.5 <= p.position[0] <= 0.5:
                        hits += 1
                        # Ландау смещается вместе с толщиной детектора
                        mpv_val = (det_thick_mm / 15.0) * 3.0
                        e_dep = moyal.rvs(loc=mpv_val, scale=0.4) 
                        energies_muon.append(e_dep)
                        
                if p.position[2] < -1.0:
                    p.is_active = False
                    
            if p.position[2] >= -1.0:
                active_positions.append([p.position[0], p.position[2]])
                active_energies.append(max(p.kinetic_energy, 10))
            
        if active_positions:
            scatter.set_offsets(np.array(active_positions))
            scatter.set_array(np.array(active_energies))
            
        counter_text.set_text(f"Поймано мюонов: {hits}")

        n_bg = np.random.poisson(bg_rate_hz * dt)
        if n_bg > 0:
            energies_bg.extend(np.random.exponential(scale=1.0, size=n_bg))

        ax2.clear()
        ax2.set_facecolor('#1e1e1e')
        ax2.set_xlim(0, 20)
        ax2.set_yscale('log')
        ax2.set_ylim(bottom=0.5)
        ax2.set_xlabel('Энерговыделение, МэВ', color='white')
        ax2.set_ylabel('Количество событий', color='white')
        ax2.set_title('Живой спектр детектора', color='white', fontsize=14)
        ax2.tick_params(colors='white')
        ax2.grid(True, alpha=0.2)
        
        if len(energies_bg) > 0:
            ax2.hist(energies_bg, bins=hist_bins, color='tomato', alpha=0.7, label='Фон помещения')
        if len(energies_muon) > 0:
            ax2.hist(energies_muon, bins=hist_bins, histtype='step', color='dodgerblue', linewidth=2.5, label='Мюоны')
            
        ax2.axvline(x=threshold_mev, color='yellow', linestyle='--', label=f'Порог ({threshold_mev:.1f} МэВ)')
        
        leg = ax2.legend(loc='upper right', facecolor='#1e1e1e', edgecolor='none')
        for text in leg.get_texts():
            text.set_color('white')
            
        return scatter, counter_text

    anim = FuncAnimation(fig, update, frames=total_frames, init_func=init, blit=False)
    gif_path = "muon_simulation_live.gif"
    writer = PillowWriter(fps=fps)
    anim.save(gif_path, writer=writer)
    plt.close(fig)
    return gif_path

# ==========================================
# МОДУЛИ ВКЛАДОК
# ==========================================

def render_attenuation_tab(det_side_mm, det_thick_mm, bg_intensity):
    st.header("Ослабление потока (Затухание по этажам)")
    
    st.info(f"🔎 **Стратегия поиска:** Сейчас уровень фона составляет **{bg_intensity:.1f}x**. "
            "Симуляция покажет, можно ли выделить сигнал затухания мюонов в таких условиях.")
    
    # --- САЙДБАР (Локальные параметры только для этой вкладки) ---
    st.sidebar.header("1. Параметры среды")
    n_layers = st.sidebar.slider("Количество слоев", 1, 10, 5)
    layer_thickness = st.sidebar.number_input("Толщина одного слоя (м)", 0.01, 2.0, 0.30, 0.05)
    
    st.sidebar.header("2. Материал")
    material_name = st.sidebar.text_input("Название", "Бетон (Concrete)")
    material_density = st.sidebar.number_input("Плотность (г/см³)", 0.1, 20.0, 2.4, 0.1)
    custom_material = MaterialLayer(material_name, layer_thickness, material_density, 0.5, 135.0)

    st.sidebar.header("3. Настройки расчета")
    duration = st.sidebar.slider("Длительность видео (сек)", 5, 120, 10, 5)
    n_muons = st.sidebar.number_input("Мюонов для Монте-Карло", 10_000, 2_000_000, 1_000_000, 100_000)

    # --- ГЛАВНЫЙ ЭКРАН ---
    
    # БЛОК 1: АНИМАЦИЯ
    st.subheader("Визуализация пролета частиц")
    if st.button("🎥 Сгенерировать анимацию", type="primary"):
        with st.spinner(f"Рендеринг физики для '{material_name}'..."):
            # Передаем толщину и интенсивность фона в анимацию
            gif_path = generate_animation_gif(n_layers, duration, custom_material, det_thick_mm, bg_intensity)
            st.image(gif_path, use_container_width=True)
            
    st.markdown("---")
    
    # БЛОК 2: НАУЧНЫЙ ГРАФИК PLOTLY
    st.subheader("Монте-Карло: График затухания")
    
    if st.button("📊 Рассчитать кривую затухания потока"):
        with st.spinner(f"Прогоняем {n_muons} частиц через {n_layers} слоев..."):
            # 1. Инициализация векторизованного ядра
            gen_muon = CosmicMuonGenerator(e_min=100.0, e_max=100000.0)
            # Частота фона теперь зависит от глобального ползунка
            gen_bg = BackgroundGenerator(rate_hz=500.0 * bg_intensity, mean_energy_mev=1.0)
            
            # В ядре используем площадь 1х1м для набора статистики, потом масштабируем
            det_thick_m = det_thick_mm / 1000.0
            detector_sim = Detector(size_x=1.0, size_y=1.0, thickness=det_thick_m, z_position=0.0)
            engine = TransportEngine(step_size_m=0.05)
            sim = Simulation(detector_sim, engine, gen_muon, gen_bg)
            
            # 2. Быстрый расчет (занимает пару секунд вместо 8 минут)
            results = sim.run_building_scan(n_primary_muons=int(n_muons), n_layers=n_layers, custom_material=custom_material)
            
            # 3. Обработка данных
            layers = sorted(results.keys())
            rates, errors, thicknesses = [], [], []
            
            # Масштабирование под реальную площадь твоего детектора
            det_side_m = det_side_mm / 1000.0
            scale_factor = det_side_m * det_side_m 
            threshold_mev = (det_thick_mm / 15.0) * 1.0 
            
            for layer in layers:
                data = results[layer]
                all_signals = np.concatenate([data['muon_signals'], data['bg_signals']])
                # Отсекаем фон по расчетному порогу
                detected = len(all_signals[all_signals > threshold_mev])
                
                rate_sim = detected / data['time']
                error_sim = np.sqrt(detected) / data['time'] if detected > 0 else 0
                
                rates.append(rate_sim * scale_factor)
                errors.append(error_sim * scale_factor)
                thicknesses.append(layer * layer_thickness)
                
            # 4. ПОСТРОЕНИЕ ГРАФИКА
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=thicknesses,
                y=rates,
                error_y=dict(type='data', array=errors, visible=True, color='#00c0f2'),
                mode='lines+markers',
                marker=dict(size=10, color='#00c0f2'),
                line=dict(width=2, color='rgba(0, 192, 242, 0.6)'),
                name='Скорость счета'
            ))
            
            fig.update_layout(
                title=dict(text=f"Детектор {det_side_mm}x{det_side_mm}x{det_thick_mm} мм | Фон: {bg_intensity}x", font=dict(size=18)),
                xaxis_title="Суммарная толщина защиты, м",
                yaxis_title="Скорость счета (Rate), Гц",
                template="plotly_dark",
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)'
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # 5. ВЫВОД РЕЗУЛЬТАТОВ
            st.info(
                f"**Прогноз:** На крыше ожидается "
                f"**{rates[0]*60:.1f} мюонов/мин**. В подвале — "
                f"**{rates[-1]*60:.1f} мюонов/мин**. "
            )
            
            # Аналитика для защиты
            snr_approx = rates[0] / (bg_intensity * 0.5) # Грубая оценка SNR
            if snr_approx < 0.2:
                st.error("⚠️ **Внимание:** Фон слишком высок. Разница между этажами тонет в статистической ошибке!")
            else:
                st.success("✅ **Успех:** Соотношение сигнал/шум позволяет зафиксировать затухание потока.")

def render_calibration_tab(det_side_mm, det_thick_mm, bg_intensity):
    st.header("Калибровка детектора (Спектр и Фон)")
    
    # Физический расчет положения пика
    thickness_cm = det_thick_mm / 10.0
    approx_mpv = thickness_cm * 2.0 
    
    st.markdown(
        f"Моделирование для сцинтиллятора **{det_side_mm}×{det_side_mm}×{det_thick_mm} мм**. "
        f"Наиболее вероятная потеря энергии (MPV): **~{approx_mpv:.1f} МэВ**.\n\n"
        f"*(Уровень радиационного фона **{bg_intensity}x** задается в глобальных настройках слева)*"
    )
    
    # --- САЙДБАР (Только локальные настройки) ---
    st.sidebar.header("Настройки дискриминатора")
    threshold = st.sidebar.slider("Порог дискриминации (МэВ)", 0.0, 10.0, 2.8, 0.1)

    # --- ГЕНЕРАЦИЯ ДАННЫХ ---
    np.random.seed(42)
    
    # Количество событий фона теперь зависит ТОЛЬКО от глобального ползунка bg_intensity
    n_bg = int(50000 * bg_intensity)
    
    # Защита от нулевого фона (чтобы логарифм и деление не сломались)
    if n_bg == 0:
        n_bg = 1 
        
    bg_events = np.random.exponential(scale=0.45, size=n_bg)
    
    # Мюоны (фиксированное количество для наглядности)
    norm_part = np.random.normal(loc=approx_mpv - 0.6, scale=0.2 + (thickness_cm*0.05), size=5000)
    lognorm_part = np.random.lognormal(mean=-0.5, sigma=0.8, size=5000)
    muon_events = norm_part + lognorm_part

    # --- РАСЧЕТ SNR (Сигнал/Шум) ---
    signal_above = np.sum(muon_events >= threshold)
    noise_above = np.sum(bg_events >= threshold)
    snr = signal_above / noise_above if noise_above > 0 else float('inf')

    # --- ГРАФИК ---
    fig = go.Figure()
    fig.add_trace(go.Histogram(x=bg_events, xbins=dict(start=0, end=10, size=0.1),
                               marker_color='tomato', name='Гамма-фон', opacity=0.7))
    fig.add_trace(go.Histogram(x=muon_events, xbins=dict(start=0, end=10, size=0.1),
                               marker_color='dodgerblue', name='Мюоны', opacity=0.7))

    fig.update_layout(
        barmode='overlay', 
        template="plotly_dark",
        xaxis_title="Энергия, МэВ", 
        yaxis_title="События (Log Scale)",
        xaxis=dict(range=[0, 10]),
        # Жестко фиксируем ось Y от 1 (10^0) до 1 000 000 (10^6)
        yaxis=dict(type="log", range=[0, 6], fixedrange=True),
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        margin=dict(l=20, r=20, t=40, b=20),
        legend=dict(x=0.70, y=0.95, bgcolor='rgba(0,0,0,0.5)')
    )
    
    fig.add_vline(x=threshold, line_dash="dash", line_color="yellow", line_width=3)
    st.plotly_chart(fig, use_container_width=True)

    # --- МЕТРИКИ ---
    c1, c2, c3 = st.columns(3)
    c1.metric("Отсеяно шума", f"{np.sum(bg_events < threshold)/n_bg*100:.2f}%")
    c2.metric("Сохранено мюонов", f"{signal_above/5000*100:.2f}%")
    c3.metric("Чистота сигнала (SNR)", f"{snr:.2f}", 
              delta="Плохо" if snr < 1 else "Хорошо", delta_color="inverse")

    if snr < 1:
        st.error(f"**Вердикт симуляции:** При текущем фоне ({bg_intensity}x) сигнал мюонов тонет в шуме. "
                 "Требуется снижение фона (ищите другое место) или увеличение толщины детектора.")
def render_poisson_tab():
    st.header("Статистика прилета (Распределение Пуассона)")
    st.markdown(
        "Космические лучи падают на Землю случайно и независимо друг от друга. "
        "По законам статистики, если количество событий подчиняется **распределению Пуассона**, "
        "то время между последовательными срабатываниями детектора (интервалы $\Delta t$) "
        "должно описываться **экспоненциальным законом**: $f(t) = \lambda e^{-\lambda t}$."
    )
    
    # --- ДИНАМИЧЕСКИЙ САЙДБАР ---
    st.sidebar.header("Параметры потока")
    
    # Скорость счета для нашего маленького детектора (30х30 мм) довольно низкая
    rate_hz = st.sidebar.slider("Средняя частота (Rate, Гц)", min_value=0.01, max_value=5.00, value=0.05, step=0.01)
    
    # Время сбора данных в часах
    sim_time_hours = st.sidebar.number_input("Время сбора данных (часов)", min_value=1, max_value=720, value=24, step=12)
    
    st.sidebar.header("Отображение")
    show_theory = st.sidebar.checkbox("Наложить теоретическую кривую", value=True)
    log_y_scale = st.sidebar.checkbox("Логарифмическая шкала Y", value=False, 
                                      help="Экспонента на логарифмической шкале выглядит как прямая линия!")

    # --- МАТЕМАТИКА И ГЕНЕРАЦИЯ ---
    sim_time_sec = sim_time_hours * 3600
    expected_events = rate_hz * sim_time_sec
    
    # Генерируем реальное количество пойманных частиц по Пуассону
    # Фиксируем seed не будем, пусть при каждом пересчете будут новые случайные флуктуации
    actual_events = np.random.poisson(expected_events)
    
    if actual_events == 0:
        st.warning("За выбранное время детектор не поймал ни одной частицы. Увеличьте время или Rate.")
        return
        
    # Защита от переполнения памяти браузера (если пользователь выкрутит ползунки на максимум)
    plot_events = min(actual_events, 200000)
    
    with st.spinner("Анализ временных интервалов..."):
        # Генерируем интервалы времени между событиями (Экспоненциальное распределение)
        # scale в numpy это 1/lambda (то есть среднее время между событиями)
        intervals = np.random.exponential(scale=1.0/rate_hz, size=plot_events)
        
        # --- ПОСТРОЕНИЕ ГРАФИКА PLOTLY ---
        fig = go.Figure()

        # Гистограмма сгенерированных интервалов (нормализованная как плотность вероятности)
        fig.add_trace(go.Histogram(
            x=intervals,
            histnorm='probability density',
            nbinsx=100,
            marker_color='mediumpurple',
            name='Эксперимент (Монте-Карло)',
            opacity=0.7
        ))

        # Наложение идеальной теоретической кривой
        if show_theory:
            t_max = np.max(intervals)
            t_arr = np.linspace(0, t_max, 500)
            # f(t) = lambda * exp(-lambda * t)
            pdf_arr = rate_hz * np.exp(-rate_hz * t_arr)
            
            fig.add_trace(go.Scatter(
                x=t_arr, 
                y=pdf_arr, 
                mode='lines', 
                line=dict(color='yellow', width=3),
                name='Теория: $\lambda e^{-\lambda t}$'
            ))

        # Настройка отображения осей
        yaxis_type = "log" if log_y_scale else "linear"
        
        fig.update_layout(
            barmode='overlay',
            xaxis_title="Интервал времени между событиями Δt, секунды",
            yaxis_title="Плотность вероятности",
            yaxis=dict(type=yaxis_type),
            template="plotly_dark",
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            margin=dict(l=20, r=20, t=40, b=20),
            legend=dict(x=0.7, y=0.95, bgcolor='rgba(0,0,0,0.5)')
        )

        st.plotly_chart(fig, use_container_width=True)

        # --- АНАЛИТИКА (Метрики) ---
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric(label="Всего зарегистрировано мюонов", value=f"{actual_events:,}".replace(',', ' '))
        with col2:
            st.metric(label="Среднее время ожидания (Теория)", value=f"{1.0/rate_hz:.2f} сек")
        with col3:
            st.metric(label="Реальное среднее Δt (Эксперимент)", value=f"{np.mean(intervals):.2f} сек")

def render_barometric_tab():
    st.header("Барометрический эффект")
    st.markdown(
        "Атмосфера Земли работает как гигантский поглотитель космических лучей. "
        "При повышении атмосферного давления масса воздуха над детектором увеличивается, "
        "что приводит к **снижению скорости счета мюонов**. Эта зависимость доказывает, "
        "что мы регистрируем именно частицы из космоса, а не локальный земной фон."
    )
    
    # --- ЛОКАЛЬНЫЙ САЙДБАР ---
    st.sidebar.header("Метеоусловия (Модель циклона)")
    
    # Коэффициент беты для мюонов обычно около 0.15 - 0.20 %/мбар
    beta_coef = st.sidebar.slider("Барометрический коэфф. (%/мбар)", 0.05, 0.40, 0.15, 0.01)
    
    # Моделируем прохождение глубокого циклона
    pressure_drop = st.sidebar.slider("Глубина циклона (мбар)", 10.0, 80.0, 40.0, 5.0)
    days = st.sidebar.slider("Длительность наблюдений (дней)", 1, 14, 7)
    
    base_rate = st.sidebar.number_input("Базовая скорость счета (Гц)", 0.01, 10.0, 0.5, step=0.1)

    # --- ГЕНЕРАЦИЯ ДАННЫХ ---
    # Генерируем массив времени (в часах)
    hours = np.arange(0, days * 24)
    
    # Моделируем давление (Базовое давление 1013 мбар)
    # Используем перевернутый Гауссиан для имитации прохождения циклона (падение давления)
    center_hour = (days * 24) / 2
    width = (days * 24) / 6
    pressure = 1013.0 - pressure_drop * np.exp(-0.5 * ((hours - center_hour) / width)**2)
    
    # Расчет идеальной скорости счета по физическому закону: I = I0 * exp(-beta * dP)
    # Переводим бету из % в абсолютные значения (делим на 100)
    beta_abs = beta_coef / 100.0
    ideal_rate = base_rate * np.exp(-beta_abs * (pressure - 1013.0))
    
    # Накладываем реальную статистику Пуассона (случайные флуктуации)
    # Считаем количество частиц за 1 час (rate * 3600)
    expected_counts_per_hour = ideal_rate * 3600
    actual_counts_per_hour = np.random.poisson(expected_counts_per_hour)
    
    # Переводим обратно в Гц для графика
    actual_rate = actual_counts_per_hour / 3600.0

    # --- ПОСТРОЕНИЕ ГРАФИКА (Две оси Y) ---
    # Создаем фигуру с дополнительной осью Y
    fig = make_subplots(specs=[[{"secondary_y": True}]])

    # Добавляем график атмосферного давления (правая ось Y)
    fig.add_trace(
        go.Scatter(x=hours, y=pressure, name="Атмосферное давление", 
                   mode="lines", line=dict(color="tomato", width=3)),
        secondary_y=True,
    )

    # Добавляем график скорости счета мюонов (левая ось Y)
    fig.add_trace(
        go.Scatter(x=hours, y=actual_rate, name="Скорость счета мюонов", 
                   mode="markers+lines", marker=dict(size=4, color="dodgerblue"),
                   line=dict(width=1, color="rgba(30, 144, 255, 0.4)")),
        secondary_y=False,
    )

    # Настраиваем внешний вид осей и графиков
    fig.update_layout(
        title=dict(text="Антикорреляция потока мюонов и давления", font=dict(size=18)),
        xaxis_title="Время наблюдений, часы",
        template="plotly_dark",
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        hovermode="x unified",
        legend=dict(x=0.01, y=0.99, bgcolor='rgba(0,0,0,0.5)')
    )

    # Названия для осей Y
    fig.update_yaxes(title_text="<b>Скорость счета</b> (Гц)", secondary_y=False, color="dodgerblue")
    fig.update_yaxes(title_text="<b>Давление</b> (мбар)", secondary_y=True, color="tomato", autorange="reversed")

    st.plotly_chart(fig, use_container_width=True)

    # --- ВЫВОД МЕТРИК ---
    st.markdown("### Анализ данных")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric(label="Макс. падение давления", value=f"-{pressure_drop:.1f} мбар")
    with col2:
        # Считаем насколько в процентах вырос поток мюонов
        max_rate_increase = ((np.max(ideal_rate) - base_rate) / base_rate) * 100
        st.metric(label="Рост потока мюонов", value=f"+{max_rate_increase:.2f} %", delta="Ожидаемо", delta_color="normal")
    with col3:
        # Считаем коэффициент корреляции Пирсона между давлением и счетом
        correlation = np.corrcoef(pressure, actual_rate)[0, 1]
        st.metric(label="Корреляция (Пирсон)", value=f"{correlation:.2f}", 
                  help="Значение близкое к -1 означает строгую обратную зависимость.")

    if correlation < -0.7:
        st.success("✅ **Вывод:** Наблюдается сильная обратная корреляция. Барометрический эффект успешно зарегистрирован.")
    else:
        st.warning("⚠️ **Внимание:** Флуктуации Пуассона слишком велики. Увеличьте базовый Rate или возьмите более глубокий циклон.")
# ==========================================
# ТОЧКА ВХОДА (МЕНЕДЖЕР МАРШРУТИЗАЦИИ)
# ==========================================
def main():
    st.sidebar.title("🔬 Навигация")
    
    # 1. Выбор режима работы
    menu_options = [
        "1. Ослабление потока", 
        "2. Калибровка детектора", 
        "3. Статистика прилета", 
        "4. Барометрический эффект"
    ]
    choice = st.sidebar.radio("Выберите режим:", menu_options)
    st.sidebar.markdown("---") 
    
    # 2. ГЛОБАЛЬНЫЕ ПАРАМЕТРЫ ДЕТЕКТОРА (видны всегда)
    st.sidebar.header("Параметры детектора")
    det_side_mm = st.sidebar.slider(
        "Ширина/Длина (мм)", 
        min_value=10, max_value=200, value=30, step=10,
        help="Размер кристалла в плане. Влияет на геометрический фактор (сколько мюонов поймаем)."
    )
    det_thick_mm = st.sidebar.slider(
        "Толщина (мм)", 
        min_value=5, max_value=100, value=15, step=1,
        help="Толщина сцинтиллятора. Влияет на количество выделенной энергии (сдвигает пик Ландау)."
    )
    
    st.sidebar.markdown("---")
    
    # 3. ГЛОБАЛЬНЫЕ ПАРАМЕТРЫ СРЕДЫ (твой «дозиметр»)
    st.sidebar.header("Окружающая среда")
    bg_intensity = st.sidebar.slider(
        "Уровень фона (Дозиметр)", 
        min_value=0.1, max_value=10.0, value=1.0, step=0.1,
        help="1.0 - стандартный фон помещения. < 1.0 - поиск чистого места (лес). > 1.0 - сильное излучение стен."
    )
    
    st.sidebar.markdown("---")

    # 4. МАРШРУТИЗАЦИЯ (Передача параметров в функции вкладок)
    # Убедись, что в самих функциях (render_...) добавлены соответствующие аргументы в скобках!
    if choice == menu_options[0]:
        render_attenuation_tab(det_side_mm, det_thick_mm, bg_intensity)
        
    elif choice == menu_options[1]:
        render_calibration_tab(det_side_mm, det_thick_mm, bg_intensity)
        
    elif choice == menu_options[2]:
        # Для статистики Пуассона пока оставим без параметров, 
        # но если захочешь привязать Rate к площади детектора — можно добавить и сюда.
        render_poisson_tab() 
        
    elif choice == menu_options[3]:
        render_barometric_tab()

# Не забудь про вызов в самом конце файла
if __name__ == "__main__":
    main()