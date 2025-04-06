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
        .select("meal_name, meal_ingredients, nutrition_total, meal_type")
        .eq("user_id", userId)
        .order("meal_type", { ascending: true });

      if (mealError) {
        setError(mealError.message);
      } else {
        const cleanedMeals = mealData.map((meal) => ({
          ...meal,
          meal_ingredients: cleanIngredientNames(meal.meal_ingredients),
          nutrition: parseNutrition(meal.nutrition_total),
        }));
        setMeals(cleanedMeals);
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
      mealRefs[type].current.scrollBy({ left: -380, behavior: "smooth" });
    }
  };

  const scrollRight = (type) => {
    if (mealRefs[type]?.current) {
      mealRefs[type].current.scrollBy({ left: 380, behavior: "smooth" });
    }
  };

  return (
    <div className="min-h-screen bg-gradient-to-b from-white to-gray-100 p-8 flex flex-col items-center">
      <h2 className="text-4xl font-bold text-gray-900 text-center mb-6">
        Your Standard Diet Plan
      </h2>
      <p className="text-md text-gray-500 text-center mb-10 max-w-2xl font-light">
        A structured meal plan designed to provide balanced nutrition for a healthy lifestyle.
        Follow these meal recommendations to meet your daily dietary needs.
      </p>

      {["breakfast", "lunch", "dinner"].map((type) => (
        <div key={type} className="w-[80%] max-w-[1400px] text-center mb-14">
          <h3 className="text-3xl font-bold text-blue-600 mb-6 capitalize">
            {type}
          </h3>

          <div className="relative">
            <button
              className="absolute left-2 top-1/2 transform -translate-y-1/2 p-4 bg-green-600 text-white rounded-full shadow-lg hover:bg-green-700 transition z-10"
              onClick={() => scrollLeft(type)}
            >
              <FaArrowLeft size={20} />
            </button>

            <div
              ref={mealRefs[type]}
              className="flex gap-6 overflow-x-auto snap-x snap-mandatory scroll-smooth px-8 hide-scrollbar"
              style={{ scrollPaddingLeft: "8px" }} // Ensures first card is fully visible
            >
              {categorizedMeals[type].map((meal) => (
                <div
                  key={meal.id}
                  className="w-[380px] min-w-[380px] bg-white shadow-xl rounded-2xl p-6 snap-center border border-gray-300"
                >
                  <h3 className="text-xl font-semibold text-gray-800 mb-3">
                    {meal.meal_name}
                  </h3>
                  <p className="text-gray-600 mb-4">
                    <span className="font-bold">Ingredients:</span> {meal.meal_ingredients}
                  </p>

                  {/* Render only available nutrition values */}
                  {Object.entries(meal.nutrition).map(([key, value]) =>
                    value !== "N/A" && value !== "0mg" && value !== "0g" ? (
                      <p className="text-gray-700 text-lg" key={key}>
                        <span className="font-bold capitalize">{key}:</span> {value}
                      </p>
                    ) : null
                  )}
                </div>
              ))}
            </div>

            <button
              className="absolute right-2 top-1/2 transform -translate-y-1/2 p-4 bg-green-600 text-white rounded-full shadow-lg hover:bg-green-700 transition z-10"
              onClick={() => scrollRight(type)}
            >
              <FaArrowRight size={20} />
            </button>
          </div>
        </div>
      ))}
    </div>
  );
};

// 🛠 Function to parse nutrition values correctly
const parseNutrition = (nutrition) => {
  try {
    const parsed = typeof nutrition === "string" ? JSON.parse(nutrition) : nutrition;
    if (!parsed) return {};

    return {
      Protein: parsed["protein"] ? `${parsed["protein"]} g` : "N/A",
      Carbs: parsed["carbohydrate"] ? `${parsed["carbohydrate"]} g` : "N/A",
      Fats: parsed["total_fat"] ? `${parsed["total_fat"]} g` : "N/A",
      Iron: parsed["iron_mg"] ? `${parsed["iron_mg"]} mg` : "N/A",
    };
  } catch (error) {
    console.error("Error parsing nutrition:", error);
    return {};
  }
};

// 🛠 Function to clean ingredient names (handle plain comma-separated string)
const cleanIngredientNames = (ingredients) => {
  if (!ingredients || typeof ingredients !== "string") {
    return "Unknown Ingredients";
  }
  // Split the comma-separated string and clean each item
  return ingredients
    .split(", ")
    .map((item) => item.split(" (")[0].trim()) // Remove scientific names if present
    .join(", ");
};

export default UserMeals;
