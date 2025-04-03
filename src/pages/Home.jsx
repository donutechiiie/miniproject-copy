import { useEffect, useState } from "react";
import { supabase } from "../supabaseClient"; // Adjust path if needed

export default function Home() {
  const [userData, setUserData] = useState(null);
  const [loading, setLoading] = useState(true);
  const [gradientStyle, setGradientStyle] = useState("from-blue-500 to-purple-500");

  // Function to cycle through gradients dynamically
  useEffect(() => {
    const gradients = [
      "from-blue-500 to-purple-500",
      "from-purple-500 to-pink-500",
      "from-pink-500 to-red-500",
      "from-red-500 to-yellow-500",
      "from-yellow-500 to-green-500",
      "from-green-500 to-blue-500",
    ];
    let index = 0;

    const changeGradient = () => {
      index = (index + 1) % gradients.length;
      setGradientStyle(gradients[index]);
    };

    const interval = setInterval(changeGradient, 3000); // Change gradient every 3 seconds

    return () => clearInterval(interval); // Cleanup function
  }, []);

  useEffect(() => {
    const fetchUserData = async () => {
      setLoading(true);
      const { data: authData, error: authError } = await supabase.auth.getUser();
      if (authError || !authData?.user) {
        console.error("Error fetching user:", authError?.message);
        setLoading(false);
        return;
      }

      const authUid = authData.user.id;
      const { data: userDetails, error: userError } = await supabase
        .from("UserTable")
        .select("firstname, lastname, email")
        .eq("auth_uid", authUid)
        .single();

      if (userError) {
        console.error("Error fetching user details:", userError.message);
      } else {
        setUserData(userDetails);
      }
      setLoading(false);
    };

    fetchUserData();
  }, []);

  return (
    <div className="max-w-5xl mx-auto px-6 py-12">
      {/* Dynamic Gradient Welcome Section */}
      <div
        className={`relative w-full h-48 md:h-56 lg:h-64 flex items-center justify-center rounded-lg text-center text-white shadow-lg transition-all duration-1000 bg-gradient-to-r ${gradientStyle}`}
      >
        <h1 className="text-3xl md:text-5xl font-bold tracking-wide animate-pulse">
          Welcome to Nutrify
        </h1>
      </div>

      {/* Description */}
      <div className="text-center mt-4 mb-8">
        <p className="text-lg text-gray-600">
          Your personal nutrition assistant, helping you maintain a balanced and healthy diet.
        </p>
      </div>

      {/* Loading State */}
      {loading && (
        <p className="text-center text-lg font-semibold text-gray-600">Loading user details...</p>
      )}

      {/* User Info Section */}
      {userData && (
        <div className="bg-blue-100 p-6 rounded-lg shadow-md text-center max-w-2xl mx-auto mb-8">
          <h2 className="text-2xl font-semibold text-gray-900">
            Hello, {userData.firstname || "User"} {userData.lastname || ""}
          </h2>
          <p className="text-gray-700">📧 Email: {userData.email || "Not provided"}</p>
        </div>
      )}

      {/* Navigation Guide */}
      <div className="bg-gray-100 p-6 rounded-lg shadow-md text-center max-w-3xl mx-auto mb-8">
        <h2 className="text-2xl font-semibold text-gray-900">Website Navigation Guide</h2>
        <ul className="text-gray-700 mt-3 text-left mx-auto max-w-xl">
          <li><strong>🏠 Home:</strong> View your details and website info.</li>
          <li><strong>📤 Upload Report:</strong> Upload and analyze your nutrition reports.</li>
          <li><strong>📊 Calculate BMI:</strong> Check your BMI and health status.</li>
          <li><strong>🥗 Personalized Diet:</strong> Get AI-powered meal recommendations.</li>
          <li><strong>🍽 Standard Diet:</strong> View predefined diet plans.</li>
          <li><strong>🚪 Logout:</strong> Securely sign out of your account.</li>
        </ul>
      </div>

      {/* Food Image Gallery */}
      <div className="mt-12 text-center">
        <h2 className="text-3xl font-semibold text-gray-900 mb-6">Explore Healthy Meals</h2>
        <div className="grid grid-cols-1 sm:grid-cols-2 md:grid-cols-3 gap-6">
          <img
            src="https://images.unsplash.com/photo-1593819559713-743d364eb059?w=1200&auto=format&fit=crop&q=60&ixlib=rb-4.0.3&ixid=M3wxMjA3fDB8MHxzZWFyY2h8NHx8dGVyaXlha2l8ZW58MHx8MHx8fDA%3D"
            alt="Healthy Bowl"
            className="rounded-lg shadow-md w-full h-48 object-cover"
          />
          <img
            src="https://images.unsplash.com/photo-1627308595228-9d0497edbe74?w=1200&auto=format&fit=crop&q=60&ixlib=rb-4.0.3&ixid=M3wxMjA3fDB8MHxzZWFyY2h8MTB8fHlvZ3VydCUyMGJvd2x8ZW58MHx8MHx8fDA%3D"
            alt="Fresh Salad"
            className="rounded-lg shadow-md w-full h-48 object-cover"
          />
          <img
            src="https://images.unsplash.com/photo-1541519227354-08fa5d50c44d?w=1200&auto=format&fit=crop&q=60&ixlib=rb-4.0.3&ixid=M3wxMjA3fDB8MHxzZWFyY2h8Mnx8YXZhY2FkbyUyMHRvYXN0JTIwd2l0aCUyMGVnZ3xlbnwwfHwwfHx8MA%3D%3D"
            alt="Protein Meal"
            className="rounded-lg shadow-md w-full h-48 object-cover"
          />
        </div>
      </div>
    </div>
  );
}
