import { useEffect, useState, useRef } from "react";
import { FaArrowLeft, FaArrowRight } from "react-icons/fa";
import { supabase } from "../supabaseClient"; // Adjust path if needed

const UserMeals = () => {
  const [meals, setMeals] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const mealRefs = {
    breakfast: useRef(null),
    lunch: useRef(null),
    dinner: useRef(null),
  };

  useEffect(() => {
    const fetchMeals = async () => {
      setLoading(true);
      setError(null);

      const { data: userData, error: userError } = await supabase.auth.getUser();
      if (userError || !userData?.user) {
        setError("User not logged in");
        setLoading(false);
        return;
      }

      const userId = userData.user.id;

      const { data: mealData, error: mealError } = await supabase
        .from("standard_recommendation")
        .select("*")
        .eq("user_id", userId)
        .order("meal_type", { ascending: true });

      if (mealError) {
        setError(mealError.message);
      } else {
        setMeals(mealData);
      }

      setLoading(false);
    };

    fetchMeals();
  }, []);

  if (loading)
    return <p className="text-center text-lg font-semibold text-gray-600">Loading...</p>;

  if (error)
    return <p className="text-center text-lg text-red-500">Error: {error}</p>;

  const categorizedMeals = {
    breakfast: meals.filter((meal) => meal.meal_type.toLowerCase() === "breakfast"),
    lunch: meals.filter((meal) => meal.meal_type.toLowerCase() === "lunch"),
    dinner: meals.filter((meal) => meal.meal_type.toLowerCase() === "dinner"),
  };

  const scrollLeft = (type) => {
    if (mealRefs[type]?.current) {
      mealRefs[type].current.scrollBy({ left: -300, behavior: "smooth" });
    }
  };

  const scrollRight = (type) => {
    if (mealRefs[type]?.current) {
      mealRefs[type].current.scrollBy({ left: 300, behavior: "smooth" });
    }
  };

  return (
    <div className="min-h-screen bg-gradient-to-b from-white to-gray-100 p-6 flex flex-col items-center">
      {/* Updated Heading and Slim Text */}
      <h2 className="text-4xl font-bold text-gray-900 text-center mb-4">
        Your Standard Diet Plan
      </h2>
      <p className="text-md text-gray-500 text-center mb-8 max-w-2xl font-light">
        A structured meal plan designed to provide balanced nutrition for a healthy lifestyle. 
        Follow these meal recommendations to meet your daily dietary needs.
      </p>

      {["breakfast", "lunch", "dinner"].map((type) => (
        <div key={type} className="w-full max-w-5xl text-center mb-12">
          <h3 className="text-3xl font-bold text-blue-600 mb-4 capitalize">
            {type}
          </h3>

          {/* Scrollable Meal Section */}
          <div className="relative">
            {/* Left Scroll Button */}
            <button
              className="absolute left-0 top-1/2 transform -translate-y-1/2 p-3 bg-gray-200 rounded-full shadow-md hover:bg-gray-300 transition z-10"
              onClick={() => scrollLeft(type)}
            >
              <FaArrowLeft />
            </button>

            {/* Meal Cards Container */}
            <div
              ref={mealRefs[type]}
              className="flex gap-6 overflow-x-auto snap-x snap-mandatory scroll-smooth px-4 hide-scrollbar justify-center"
            >
              {categorizedMeals[type].map((meal) => (
                <div
                  key={meal.id}
                  className="w-80 min-w-[250px] bg-white shadow-lg rounded-2xl p-5 snap-center border border-gray-200"
                >
                  <h3 className="text-lg font-semibold text-gray-800">
                    {meal.meal_name}
                  </h3>
                  <p className="text-gray-600 mt-2">
                    <span className="font-bold">Ingredients:</span> {meal.meal_ingredients}
                  </p>
                  <p className="text-gray-600 mt-2">
                    <span className="font-bold">Nutritional Info:</span> {meal.nutrition_total}
                  </p>
                  <p className="text-sm text-gray-500 mt-4">
                    Created: {new Date(meal.created_at).toLocaleDateString()}
                  </p>
                </div>
              ))}
            </div>

            {/* Right Scroll Button */}
            <button
              className="absolute right-0 top-1/2 transform -translate-y-1/2 p-3 bg-gray-200 rounded-full shadow-md hover:bg-gray-300 transition z-10"
              onClick={() => scrollRight(type)}
            >
              <FaArrowRight />
            </button>
          </div>
        </div>
      ))}
    </div>
  );
};

export default UserMeals;
