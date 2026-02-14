document.addEventListener('DOMContentLoaded', function () {
    const form = document.getElementById('predictionForm');
    const submitBtn = document.getElementById('submitBtn');
    const spinner = document.getElementById('loadingSpinner');

    form.addEventListener('submit', function (event) {
        if (!form.checkValidity()) {
            event.preventDefault();
            event.stopPropagation();
        } else {
            // Show loading state
            submitBtn.disabled = true;
            spinner.classList.remove('d-none');
            submitBtn.innerHTML = '<span class="spinner-border spinner-border-sm" role="status" aria-hidden="true"></span> Processing...';
        }

        form.classList.add('was-validated');
    });
});
