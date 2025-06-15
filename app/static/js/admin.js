document.addEventListener('DOMContentLoaded', function() {
    // Handle alert dismissals
    const closeButtons = document.querySelectorAll('.alert .close-btn');
    closeButtons.forEach(button => {
        button.addEventListener('click', function() {
            const alert = this.parentElement;
            alert.style.opacity = '0';
            setTimeout(() => {
                alert.style.display = 'none';
            }, 300);
        });
    });
    
    // Mobile sidebar toggle
    const sidebarToggle = document.querySelector('.sidebar-toggle');
    if (sidebarToggle) {
        sidebarToggle.addEventListener('click', function() {
            const sidebar = document.querySelector('.sidebar');
            sidebar.classList.toggle('active');
        });
    }
    
    // Handle filtering forms
    const filterForms = document.querySelectorAll('.filter-form');
    filterForms.forEach(form => {
        const clearButton = form.querySelector('a[href*="clear"]');
        if (clearButton) {
            clearButton.addEventListener('click', function(e) {
                e.preventDefault();
                const inputs = form.querySelectorAll('input, select');
                inputs.forEach(input => {
                    input.value = '';
                });
                form.submit();
            });
        }
    });
    
    // Initialize date pickers with current values
    const dateInputs = document.querySelectorAll('input[type="date"]');
    dateInputs.forEach(input => {
        if (!input.value && input.getAttribute('data-default') === 'today') {
            const today = new Date().toISOString().split('T')[0];
            input.value = today;
        }
    });
});
