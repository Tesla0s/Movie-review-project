from django import forms

class ReviewForm(forms.Form):
    review = forms.CharField(
        widget=forms.Textarea(attrs={'rows': 5, 'cols': 60}),
        label='Ваш отзыв о фильме'
    )
