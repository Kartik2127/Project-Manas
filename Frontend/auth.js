document.addEventListener("DOMContentLoaded", function() {
    const username = localStorage.getItem("username");
    const userType = localStorage.getItem("userType"); 
    const loginBtn = document.querySelector(".btn-login");

    if (username && loginBtn) {
        loginBtn.textContent = `Logout (${username})`;
        loginBtn.href = "#";
        
        loginBtn.addEventListener("click", function(event) {
            event.preventDefault();
            logoutUser();
        });
    }
});

function logoutUser() {
    localStorage.removeItem("username");
    localStorage.removeItem("userType");
    
    alert("You have been logged out.");
    window.location.href = "index.html";
}