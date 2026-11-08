"""
Кастомные шаблонные фильтры для Ivanna Lab.
"""

from django import template
from django.utils.html import format_html
from django.utils import timezone
from django.db.models import Count, Avg, Max, Min, Q, Sum
from datetime import datetime, timedelta
import json
import math
import re

register = template.Library()

# ============================================================================
# ФИЛЬТРЫ ДЛЯ МОДЕЛЕЙ
# ============================================================================

@register.filter
def model_type_class(model_type):
    """CSS класс для типа модели"""
    class_map = {
        'QWEN': 'model-qwen',
        'LLAMA': 'model-llama',
        'MISTRAL': 'model-mistral',
        'GEMMA': 'model-gemma',
        'CUSTOM': 'model-custom',
    }
    return class_map.get(model_type, 'model-custom')

@register.filter
@register.filter
def status_class(status):
    class_map = {
        'PENDING': 'bg-secondary',
        'PREPARING': 'bg-info',
        'TRAINING': 'bg-primary',
        'RUNNING': 'bg-primary',
        'COMPLETED': 'bg-success',
        'FAILED': 'bg-danger',
        'CANCELLED': 'bg-warning',
        'VALIDATING': 'bg-info',
        'DOWNLOADING': 'bg-info',
        'LOADING': 'bg-info',
        'LOADED': 'bg-success',
        'ERROR': 'bg-danger',
    }
    return class_map.get(status, 'bg-secondary')

@register.filter
def status_icon(status):
    """Иконка для статуса"""
    icon_map = {
        'LOADED': 'bi-check-circle-fill',
        'DOWNLOADING': 'bi-cloud-download',
        'ERROR': 'bi-x-circle-fill',
        'MERGED': 'bi-arrow-left-right',
        'ARCHIVED': 'bi-archive',
        'PENDING': 'bi-clock',
    }
    return icon_map.get(status, 'bi-question-circle')

@register.filter
def safety_class(score):
    """CSS класс для оценки безопасности"""
    if score is None:
        return 'status-pending'
    elif score >= 0.8:
        return 'bg-success'
    elif score >= 0.6:
        return 'bg-warning'
    else:
        return 'bg-danger'

@register.filter
def safety_icon(score):
    """Иконка для безопасности"""
    if score is None:
        return 'bi-question-circle'
    elif score >= 0.8:
        return 'bi-shield-check'
    elif score >= 0.6:
        return 'bi-shield-exclamation'
    else:
        return 'bi-shield-x'

@register.filter
def quantization_class(quantization):
    """CSS класс для типа квантизации"""
    class_map = {
        'float32': 'bg-info',
        'float16': 'bg-primary',
        'bfloat16': 'bg-primary',
        'int8': 'bg-warning',
        'int4': 'bg-warning',
        'gptq': 'bg-success',
        'awq': 'bg-success',
    }
    return class_map.get(quantization, 'bg-secondary')

@register.filter
def get_metrics_value(metrics_dict, key):
    """Получить значение из словаря метрик"""
    return metrics_dict.get(key, 0.0)

@register.filter
def get_index(lst, i):
    """
    Возвращает элемент списка по индексу.
    Использование: {{ my_list|get_index:forloop.counter0 }}
    """
    try:
        return lst[i]
    except (IndexError, TypeError, ValueError):
        return None

@register.filter
def index_of_max(values):
    """
    Возвращает индекс первого максимального элемента в списке.
    Если список пуст или некорректен, возвращает -1.
    Пример использования: {{ scores|index_of_max }} -> 2 (если scores[2] максимальный)
    """
    try:
        # Преобразуем элементы в числа (могут быть строки)
        numeric = [float(v) for v in values if v is not None]
        if not numeric:
            return -1
        max_val = max(numeric)
        # Находим индекс первого вхождения максимума в исходном списке
        for i, v in enumerate(values):
            try:
                if float(v) == max_val:
                    return i
            except (TypeError, ValueError):
                continue
        return -1
    except (TypeError, ValueError):
        return -1

@register.filter
def index_of_min(values):
    try:
        numeric = [float(v) for v in values if v is not None]
        if not numeric:
            return -1
        min_val = min(numeric)
        for i, v in enumerate(values):
            try:
                if float(v) == min_val:
                    return i
            except (TypeError, ValueError):
                continue
        return -1
    except (TypeError, ValueError):
        return -1
        
# ============================================================================
# ФОРМАТИРОВАНИЕ ЧИСЕЛ
# ============================================================================

@register.filter
def humanize_number(value):
    """Форматирование больших чисел с разделителями"""
    if value is None:
        return "0"
    
    try:
        value = int(value)
        return f"{value:,}".replace(",", " ")
    except (ValueError, TypeError):
        return str(value)

@register.filter
def humanize_size_gb(value):
    """Форматирование размера в GB с 2 знаками после запятой"""
    if value is None:
        return "0 GB"
    
    try:
        size = float(value)
        return f"{size:,.2f} GB".replace(",", " ")
    except (ValueError, TypeError):
        return f"{value} GB"

@register.filter
def humanize_size_mb(value):
    """Форматирование размера в MB"""
    if value is None:
        return "0 MB"
    
    try:
        size = float(value)
        return f"{size:,.0f} MB".replace(",", " ")
    except (ValueError, TypeError):
        return f"{value} MB"

@register.filter
def humanize_speed(value):
    """Форматирование скорости (токены/сек)"""
    if value is None:
        return "0"
    
    try:
        speed = float(value)
        if speed >= 1000:
            return f"{speed/1000:.1f}K/sec"
        return f"{speed:.1f}/sec"
    except (ValueError, TypeError):
        return str(value)

@register.filter
def percentage(value, decimals=2):
    """Форматирование процентов"""
    if value is None:
        return "0%"
    
    try:
        num = float(value)
        return f"{num:.{decimals}f}%"
    except (ValueError, TypeError):
        return f"{value}%"

@register.filter
def float_format(value, decimals=3):
    """Форматирование float с заданной точностью"""
    if value is None:
        return "0"
    
    try:
        num = float(value)
        return f"{num:.{decimals}f}"
    except (ValueError, TypeError):
        return str(value)

@register.filter
def scientific_notation(value, precision=2):
    """Научная нотация для очень больших/маленьких чисел"""
    if value is None:
        return "0"
    
    try:
        num = float(value)
        if abs(num) >= 1e6 or (abs(num) < 1e-3 and num != 0):
            return f"{num:.{precision}e}"
        return float_format(num, precision)
    except (ValueError, TypeError):
        return str(value)

# ============================================================================
# МАТЕМАТИЧЕСКИЕ ОПЕРАЦИИ
# ============================================================================

@register.filter
def add(value, arg):
    """Сложение: value + arg"""
    try:
        return float(value) + float(arg)
    except (ValueError, TypeError):
        return value

@register.filter
def subtract(value, arg):
    """Вычитание: value - arg"""
    try:
        return float(value) - float(arg)
    except (ValueError, TypeError):
        return value

@register.filter
def multiply(value, arg):
    """Умножение: value * arg"""
    try:
        return float(value) * float(arg)
    except (ValueError, TypeError):
        return value

@register.filter
def divide(value, arg):
    """Деление: value / arg"""
    try:
        arg = float(arg)
        if arg == 0:
            return 0
        return float(value) / arg
    except (ValueError, TypeError):
        return 0

@register.filter
def modulo(value, arg):
    """Остаток от деления: value % arg"""
    try:
        return int(value) % int(arg)
    except (ValueError, TypeError):
        return 0

@register.filter
def round_number(value, precision=0):
    """Округление числа"""
    try:
        return round(float(value), precision)
    except (ValueError, TypeError):
        return value

@register.filter
def ceil(value):
    """Округление вверх"""
    try:
        return math.ceil(float(value))
    except (ValueError, TypeError):
        return value

@register.filter
def floor(value):
    """Округление вниз"""
    try:
        return math.floor(float(value))
    except (ValueError, TypeError):
        return value

@register.filter
def sqrt(value):
    """Квадратный корень"""
    try:
        return math.sqrt(float(value))
    except (ValueError, TypeError):
        return 0

@register.filter
def power(value, exponent):
    """Возведение в степень: value ^ exponent"""
    try:
        return math.pow(float(value), float(exponent))
    except (ValueError, TypeError):
        return value

@register.filter
def log10(value):
    """Десятичный логарифм"""
    try:
        return math.log10(float(value))
    except (ValueError, TypeError):
        return 0

# ============================================================================
# ФИЛЬТРЫ ДЛЯ ДАТ И ВРЕМЕНИ
# ============================================================================

@register.filter
def time_ago(value):
    """Время в относительном формате (3 часа назад)"""
    if not value:
        return "никогда"
    
    now = timezone.now()
    if isinstance(value, str):
        try:
            value = datetime.fromisoformat(value.replace('Z', '+00:00'))
        except (ValueError, TypeError):
            return str(value)
    
    delta = now - value
    
    if delta.days > 365:
        years = delta.days // 365
        return f"{years} год назад" if years == 1 else f"{years} лет назад"
    elif delta.days > 30:
        months = delta.days // 30
        return f"{months} месяц назад" if months == 1 else f"{months} месяцев назад"
    elif delta.days > 0:
        return f"{delta.days} дней назад"
    elif delta.seconds > 3600:
        hours = delta.seconds // 3600
        return f"{hours} час назад" if hours == 1 else f"{hours} часов назад"
    elif delta.seconds > 60:
        minutes = delta.seconds // 60
        return f"{minutes} мин назад"
    else:
        return "только что"

@register.filter
def short_date(value):
    """Короткий формат даты: 12 янв"""
    if not value:
        return ""
    
    months = ['янв', 'фев', 'мар', 'апр', 'май', 'июн', 
              'июл', 'авг', 'сен', 'окт', 'ноя', 'дек']
    
    try:
        return f"{value.day} {months[value.month - 1]}"
    except (AttributeError, IndexError):
        return str(value)

@register.filter
def datetime_format(value, format_str="%d.%m.%Y %H:%M"):
    """Кастомный формат даты"""
    if not value:
        return ""
    
    try:
        return value.strftime(format_str)
    except AttributeError:
        return str(value)

@register.filter
def duration_format(seconds):
    """Форматирование длительности: 1h 30m"""
    if not seconds:
        return "0s"
    
    try:
        seconds = float(seconds)
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = int(seconds % 60)
        
        if hours > 0:
            return f"{hours}h {minutes}m"
        elif minutes > 0:
            return f"{minutes}m {secs}s"
        else:
            return f"{secs}s"
    except (ValueError, TypeError):
        return str(seconds)

@register.filter
def is_recent(value, days=7):
    """Проверка, была ли дата недавно (по умолчанию в течение 7 дней)"""
    if not value:
        return False
    
    try:
        now = timezone.now()
        if isinstance(value, str):
            value = datetime.fromisoformat(value.replace('Z', '+00:00'))
        
        return (now - value).days <= days
    except (ValueError, TypeError):
        return False

# ============================================================================
# РАБОТА С JSON И СЛОВАРЯМИ
# ============================================================================

@register.filter
def json_pretty(value):
    """Красивое форматирование JSON"""
    if not value:
        return ""
    
    try:
        if isinstance(value, str):
            data = json.loads(value)
        else:
            data = value
        
        return json.dumps(data, indent=2, ensure_ascii=False)
    except (json.JSONDecodeError, TypeError):
        return str(value)

@register.filter
def dict_get(value, key):
    """Получить значение из словаря по ключу"""
    try:
        return value.get(key)
    except AttributeError:
        return None

@register.filter
def dict_items(value):
    """Получить items из словаря"""
    try:
        return value.items()
    except AttributeError:
        return []

@register.filter
def dict_keys(value):
    """Получить ключи словаря"""
    try:
        return value.keys()
    except AttributeError:
        return []

@register.filter
def dict_values(value):
    """Получить значения словаря"""
    try:
        return value.values()
    except AttributeError:
        return []

@register.filter
def dict_length(value):
    """Длина словаря"""
    try:
        return len(value)
    except (AttributeError, TypeError):
        return 0

# ============================================================================
# РАБОТА СО СТРОКАМИ И ТЕКСТОМ
# ============================================================================

@register.filter
def truncate_chars(value, length=100):
    """Обрезать строку до указанной длины"""
    if not value:
        return ""
    
    value = str(value)
    if len(value) <= length:
        return value
    
    return value[:length] + "..."

@register.filter
def truncate_words(value, words=30):
    """Обрезать строку по количеству слов"""
    if not value:
        return ""
    
    word_list = str(value).split()
    if len(word_list) <= words:
        return value
    
    return " ".join(word_list[:words]) + "..."

@register.filter
def split(value, arg):
    """Разделяет строку по разделителю arg"""
    if value is None:
        return []
    return value.split(arg)
    
@register.filter
def split_string(value, delimiter=","):
    """Разделить строку по разделителю"""
    if not value:
        return []
    return str(value).split(delimiter)

@register.filter
def join_list(value, delimiter=", "):
    """Объединить список в строку"""
    if not value:
        return ""
    
    try:
        return delimiter.join(str(item) for item in value)
    except TypeError:
        return str(value)

@register.filter
def linebreaks_br(value):
    """Заменить переносы строк на <br>"""
    if not value:
        return ""
    
    from django.utils.html import escape
    return escape(value).replace('\n', '<br>')

@register.filter
def remove_whitespace(value):
    """Удалить все пробелы и переносы строк"""
    if not value:
        return ""
    
    import re
    return re.sub(r'\s+', '', str(value))

@register.filter
def highlight_json(value):
    """Подсветка JSON в HTML (простая версия)"""
    if not value:
        return ""
    
    try:
        import re
        json_str = json.dumps(value, indent=2, ensure_ascii=False)
        
        # Простая подсветка
        json_str = re.sub(r'"(.*?)":', r'<span class="text-warning">"\1"</span>:', json_str)
        json_str = re.sub(r': "(.*?)"', r': <span class="text-info">"\1"</span>', json_str)
        json_str = re.sub(r': (\d+)', r': <span class="text-success">\1</span>', json_str)
        json_str = re.sub(r': (true|false|null)', r': <span class="text-danger">\1</span>', json_str)
        
        return format_html('<pre class="bg-dark p-3 rounded">{}</pre>', json_str)
    except (TypeError, ValueError):
        return str(value)

@register.filter
def replace(value, arg):
    """
    Замена подстроки в строке.
    Использование: {{ text|replace:"старая_подстрока,новая_подстрока" }}
    """
    if not value or not arg:
        return value
    
    try:
        # Разделяем аргумент на старую и новую подстроки
        if ',' in arg:
            old_str, new_str = arg.split(',', 1)
        else:
            old_str = arg
            new_str = ''
        
        return str(value).replace(old_str, new_str)
    except (ValueError, AttributeError):
        return value

@register.filter
def replace_all(value, replacements):
    """
    Множественная замена подстрок.
    Использование: {{ text|replace_all:"старая1=новая1;старая2=новая2" }}
    """
    if not value or not replacements:
        return value
    
    try:
        result = str(value)
        # Разделяем пары замен
        pairs = replacements.split(';')
        for pair in pairs:
            if '=' in pair:
                old_str, new_str = pair.split('=', 1)
                result = result.replace(old_str, new_str)
        return result
    except (ValueError, AttributeError):
        return value

@register.filter
def escape_html(value):
    """Экранирование HTML тегов в строке"""
    if not value:
        return ""
    
    from django.utils.html import escape
    return escape(str(value))

@register.filter
def markdown_to_html(value):
    """Преобразование Markdown в HTML (упрощенное)"""
    if not value:
        return ""
    
    import re
    text = str(value)
    
    # Заголовки
    text = re.sub(r'^# (.*?)$', r'<h1>\1</h1>', text, flags=re.MULTILINE)
    text = re.sub(r'^## (.*?)$', r'<h2>\1</h2>', text, flags=re.MULTILINE)
    text = re.sub(r'^### (.*?)$', r'<h3>\1</h3>', text, flags=re.MULTILINE)
    
    # Жирный текст
    text = re.sub(r'\*\*(.*?)\*\*', r'<strong>\1</strong>', text)
    text = re.sub(r'__(.*?)__', r'<strong>\1</strong>', text)
    
    # Курсив
    text = re.sub(r'\*(.*?)\*', r'<em>\1</em>', text)
    text = re.sub(r'_(.*?)_', r'<em>\1</em>', text)
    
    # Списки
    text = re.sub(r'^\* (.*?)$', r'<li>\1</li>', text, flags=re.MULTILINE)
    lines = text.split('\n')
    in_list = False
    result_lines = []
    
    for line in lines:
        if line.startswith('<li>'):
            if not in_list:
                result_lines.append('<ul>')
                in_list = True
            result_lines.append(line)
        else:
            if in_list:
                result_lines.append('</ul>')
                in_list = False
            result_lines.append(line)
    
    if in_list:
        result_lines.append('</ul>')
    
    text = '\n'.join(result_lines)
    
    # Ссылки
    text = re.sub(r'\[([^\]]+)\]\(([^)]+)\)', r'<a href="\2" target="_blank">\1</a>', text)
    
    # Код
    text = re.sub(r'`([^`]+)`', r'<code>\1</code>', text)
    
    # Блоки кода
    text = re.sub(r'```(\w+)?\n([^`]+)```', r'<pre><code class="language-\1">\2</code></pre>', text, flags=re.DOTALL)
    
    # Переносы строк
    text = text.replace('\n\n', '</p><p>')
    text = text.replace('\n', '<br>')
    
    return f'<div class="markdown-content">{text}</div>'

@register.filter
def truncate_middle(value, length=50):
    """Обрезать строку по середине с добавлением ..."""
    if not value:
        return ""
    
    value = str(value)
    if len(value) <= length:
        return value
    
    half = length // 2
    return value[:half] + "..." + value[-half:]

@register.filter
def split_lines(value):
    """Разделить текст на строки и вернуть список"""
    if not value:
        return []
    return str(value).splitlines()

@register.filter
def count_words(value):
    """Подсчитать количество слов в тексте"""
    if not value:
        return 0
    return len(str(value).split())

@register.filter
def count_characters(value):
    """Подсчитать количество символов в тексте"""
    if not value:
        return 0
    return len(str(value))

@register.filter
def strip_punctuation(value):
    """Удалить пунктуацию из текста"""
    if not value:
        return ""
    
    import re
    return re.sub(r'[^\w\s]', '', str(value))

@register.filter
def extract_code_blocks(value):
    """Извлечь блоки кода из текста"""
    if not value:
        return []
    
    import re
    return re.findall(r'```(?:\w+)?\n([^`]+)```', str(value), re.DOTALL)

@register.filter
def chat_message_class(role):
    """CSS класс для сообщения чата по роли"""
    class_map = {
        'system': 'chat-message-system',
        'user': 'chat-message-user',
        'assistant': 'chat-message-assistant',
        'function': 'chat-message-function',
        'tool': 'chat-message-tool',
    }
    return class_map.get(role, 'chat-message-default')

@register.filter
def chat_message_icon(role):
    """Иконка для сообщения чата"""
    icon_map = {
        'system': 'bi-gear',
        'user': 'bi-person',
        'assistant': 'bi-robot',
        'function': 'bi-code',
        'tool': 'bi-wrench',
    }
    return icon_map.get(role, 'bi-question-circle')

@register.filter
def dict_get(d, key):
    """Возвращает значение из словаря по ключу."""
    return d.get(key)

# ============================================================================
# РАБОТА СО СПИСКАМИ И КОРТЕЖАМИ
# ============================================================================

@register.filter
def list_length(value):
    """Длина списка/кортежа"""
    try:
        return len(value)
    except TypeError:
        return 0

@register.filter
def get_item(value, index):
    """Получить элемент списка по индексу"""
    try:
        return value[index]
    except (TypeError, IndexError):
        return None

@register.filter
def slice_list(value, slice_str):
    """Срез списка: "start:end:step" или просто число"""
    try:
        if ':' in slice_str:
            parts = slice_str.split(':')
            start = int(parts[0]) if parts[0] else None
            end = int(parts[1]) if parts[1] else None
            step = int(parts[2]) if len(parts) > 2 and parts[2] else None
            return value[start:end:step]
        else:
            n = int(slice_str)
            return value[:n]
    except (ValueError, TypeError):
        return value

@register.filter
def filter_list(value, attr):
    """Фильтрация списка объектов по атрибуту"""
    try:
        return [item for item in value if getattr(item, attr, False)]
    except (TypeError, AttributeError):
        return []

@register.filter
def sort_by(value, attr):
    """Сортировка списка объектов по атрибуту"""
    try:
        return sorted(value, key=lambda x: getattr(x, attr, ''))
    except (TypeError, AttributeError):
        return value

@register.filter
def get_attribute(obj, attr_name):
    """Получение атрибута объекта"""
    try:
        return getattr(obj, attr_name)
    except AttributeError:
        try:
            return obj.get(attr_name)
        except (AttributeError, KeyError):
            return None

# ============================================================================
# ЛОГИЧЕСКИЕ ФИЛЬТРЫ
# ============================================================================

@register.filter
def is_none(value):
    """Проверка на None"""
    return value is None

@register.filter
def is_not_none(value):
    """Проверка, что не None"""
    return value is not None

@register.filter
def is_empty(value):
    """Проверка на пустоту"""
    if value is None:
        return True
    if isinstance(value, (list, dict, tuple, set)):
        return len(value) == 0
    if isinstance(value, str):
        return value.strip() == ""
    return False

@register.filter
def is_not_empty(value):
    """Проверка, что не пусто"""
    return not is_empty(value)

@register.filter
def contains(value, arg):
    """Проверка содержания подстроки или элемента"""
    try:
        return arg in value
    except TypeError:
        return str(arg) in str(value)

# ============================================================================
# ПРОЧИЕ ПОЛЕЗНЫЕ ФИЛЬТРЫ
# ============================================================================

@register.filter
def default_if_none(value, default):
    """Вернуть значение по умолчанию если None"""
    return value if value is not None else default

@register.filter
def default_if_empty(value, default):
    """Вернуть значение по умолчанию если пусто"""
    return value if is_not_empty(value) else default

@register.filter
def boolean_icon(value):
    """Иконка для булевого значения"""
    if value:
        return format_html('<i class="bi bi-check-circle-fill text-success"></i>')
    else:
        return format_html('<i class="bi bi-x-circle-fill text-danger"></i>')

@register.filter
def boolean_text(value):
    """Текст для булевого значения"""
    return "Да" if value else "Нет"

@register.filter
def pluralize_ru(value, forms):
    """Русская форма множественного числа"""
    # forms = "модель,модели,моделей"
    forms_list = forms.split(',')
    
    try:
        n = int(value) % 100
        n1 = n % 10
        
        if 10 < n < 20:
            return f"{value} {forms_list[2]}"
        if n1 == 1:
            return f"{value} {forms_list[0]}"
        if 2 <= n1 <= 4:
            return f"{value} {forms_list[1]}"
        return f"{value} {forms_list[2]}"
    except (ValueError, IndexError):
        return f"{value} {forms_list[0]}"

@register.filter
def progress_color(percentage):
    """Цвет для прогресс-бара по проценту"""
    try:
        perc = float(percentage)
        if perc < 30:
            return "bg-danger"
        elif perc < 70:
            return "bg-warning"
        else:
            return "bg-success"
    except (ValueError, TypeError):
        return "bg-secondary"

# ============================================================================
# ФИЛЬТРЫ ДЛЯ ИНТЕРФЕЙСА
# ============================================================================

@register.filter
def badge_color_from_value(value, value_type="model"):
    """Выбор цвета бейджа на основе значения"""
    if value_type == "model":
        colors = {
            'QWEN': 'bg-primary',
            'LLAMA': 'bg-success',
            'MISTRAL': 'bg-info',
            'GEMMA': 'bg-warning',
            'CUSTOM': 'bg-secondary',
        }
        return colors.get(value, 'bg-secondary')
    
    elif value_type == "status":
        colors = {
            'LOADED': 'bg-success',
            'DOWNLOADING': 'bg-info',
            'ERROR': 'bg-danger',
            'MERGED': 'bg-purple',
            'PENDING': 'bg-secondary',
        }
        return colors.get(value, 'bg-secondary')
    
    return 'bg-secondary'

@register.filter
def icon_from_value(value, value_type="status"):
    """Выбор иконки на основе значения"""
    icons = {
        'status': {
            'LOADED': 'bi-check-circle-fill',
            'DOWNLOADING': 'bi-cloud-download',
            'ERROR': 'bi-x-circle-fill',
            'MERGED': 'bi-arrow-left-right',
            'PENDING': 'bi-clock',
        },
        'safety': {
            'safe': 'bi-shield-check',
            'unsafe': 'bi-shield-x',
            'unknown': 'bi-shield-exclamation',
        }
    }
    
    return icons.get(value_type, {}).get(value, 'bi-question-circle')

# ============================================================================
# ОТДЕЛЬНЫЕ ФИЛЬТРЫ ДЛЯ ШАБЛОНОВ
# ============================================================================

# evaluation_task_detail.html
@register.filter
def first_item(iterable):
    """Получение первого элемента итерируемого объекта"""
    try:
        return next(iter(iterable))
    except (StopIteration, TypeError):
        return None

@register.filter
def filter_by_success(results):
    """Фильтрует результаты по успешности (не содержат ERROR)"""
    if not results:
        return 0
    
    count = 0
    for result in results:
        # Проверяем, что ответ не начинается с ERROR
        if hasattr(result, 'model_response'):
            if not result.model_response.startswith('ERROR:'):
                count += 1
    return count

@register.filter
def avg_time(results):
    """Среднее время ответов"""
    if not results:
        return 0
    
    total_time = 0
    valid_results = 0
    
    for result in results:
        if hasattr(result, 'response_time') and result.response_time > 0:
            total_time += result.response_time
            valid_results += 1
    
    return total_time / valid_results if valid_results > 0 else 0

@register.filter
def parse_log(line):
    """
    Парсит строку лога вида [timestamp] LEVEL message
    Возвращает словарь с ключами: timestamp, level, message, level_class
    """
    result = {
        'timestamp': '',
        'level': '',
        'message': line,
        'level_class': 'info'
    }
    # Регулярное выражение для формата [2026-02-23 16:44:40] INFO Starting...
    match = re.match(r'^\[(.*?)\]\s+(\w+)\s+(.*)$', line)
    if match:
        result['timestamp'] = match.group(1)
        result['level'] = match.group(2)
        result['message'] = match.group(3)
        # Определяем класс на основе уровня
        level_upper = result['level'].upper()
        if level_upper in ('ERROR', 'CRITICAL'):
            result['level_class'] = 'error'
        elif level_upper in ('WARNING', 'WARN'):
            result['level_class'] = 'warning'
        elif level_upper == 'SUCCESS':
            result['level_class'] = 'success'
        else:
            result['level_class'] = 'info'
    return result

@register.filter
def clean_text(value):
    """Убирает лишние пустые строки и обрезает пробелы."""
    if not value:
        return ''
    lines = value.splitlines()
    # Убираем пробелы в конце каждой строки
    lines = [line.rstrip() for line in lines]
    # Схлопываем несколько пустых строк в одну
    cleaned = []
    last_empty = False
    for line in lines:
        if line == '':
            if not last_empty:
                cleaned.append('')
                last_empty = True
        else:
            cleaned.append(line)
            last_empty = False
    return '\n'.join(cleaned).strip()