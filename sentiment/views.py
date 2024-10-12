from django.shortcuts import render
from django.views import View
import joblib
from .forms import ReviewForm
import nltk
import re

# Загрузка необходимых ресурсов nltk
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('punkt')

# Инициализация стоп-слов и лемматизатора
stop_words = set(nltk.corpus.stopwords.words('english'))
lemmatizer = nltk.stem.WordNetLemmatizer()

# Загрузка модели и векторайзера
model = joblib.load('sentiment_model.joblib')
vectorizer = joblib.load('tfidf_vectorizer.joblib')

# Функция предобработки текста
def preprocess_text(text):
    # Приведение к нижнему регистру
    text = text.lower()
    # Удаление HTML-тегов
    text = re.sub(r'<.*?>', '', text)
    # Удаление чисел и специальных символов
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    # Токенизация
    tokens = nltk.word_tokenize(text)
    # Удаление стоп-слов и лемматизация
    tokens = [lemmatizer.lemmatize(word) for word in tokens if word not in stop_words]
    return ' '.join(tokens)

# Представление для главной страницы
class IndexView(View):
    def get(self, request):
        form = ReviewForm()
        # Получение результатов из сессии
        sentiment = request.session.get('sentiment', None)
        rating = request.session.get('rating', None)
        context = {
            'form': form,
            'sentiment': sentiment,
            'rating': rating
        }
        return render(request, 'sentiment/index.html', context)

    def post(self, request):
        print("POST-запрос получен")  # Логирование
        form = ReviewForm(request.POST)
        if form.is_valid():
            print("Форма валидна")  # Логирование
            review_text = form.cleaned_data['review']
            print(f"Текст отзыва: {review_text}")  # Логирование
            
            try:
                # Предобработка текста
                processed_text = preprocess_text(review_text)
                print(f"Предобработанный текст: {processed_text}")  # Логирование

                # Векторизация
                vectorized_text = vectorizer.transform([processed_text])
                print(f"Векторизованный текст: {vectorized_text}")  # Логирование

                # Предсказание
                prediction = model.predict(vectorized_text)[0]
                print(f"Предсказание: {prediction}")  # Логирование

                # Оценка вероятности
                proba = model.predict_proba(vectorized_text)
                positive_prob = proba[0][1]
                print(f"Вероятность положительного отзыва: {positive_prob}")  # Логирование

                # Присвоение рейтинга
                rating = int(positive_prob * 10)
                sentiment = "Положительный" if prediction == 1 else "Отрицательный"
                print(f"Рейтинг: {rating}, Тональность: {sentiment}")  # Логирование

                # Сохранение результатов в сессии
                request.session['sentiment'] = sentiment
                request.session['rating'] = rating

                # Возвращаем результат на странице без перезагрузки
                context = {
                    'form': form,
                    'sentiment': sentiment,
                    'rating': str(rating)
                }
                print(f'CONTEXT:', end=' ')
                print(context)
                return render(request, 'sentiment/index.html', context)

            except Exception as e:
                print(f"Ошибка при обработке отзыва: {e}")  # Логирование ошибки
                # Выводим ошибку пользователю
                context = {
                    'form': form,
                    'error': "Произошла ошибка при обработке вашего отзыва."
                }
                return render(request, 'sentiment/index.html', context)

        # Если форма не валидна, выводим ошибки
        print("Форма не валидна")  # Логирование
        return render(request, 'sentiment/index.html', {'form': form})


