import { useEffect, useState, useRef } from "react";
import { FaArrowLeft, FaArrowRight } from "react-icons/fa";
import { supabase } from "../supabaseClient";

const Personalized = () => {
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
        .from("personal_recommendation")
        .select("*")
        .eq("user_id", userId)
        .order("meal_type", { ascending: true });

      if (mealError) {
        setError(mealError.message);
      } else {
        const cleanedMeals = mealData.map((meal) => ({
          ...meal,
          meal_ingredients: meal.meal_ingredients
            ?.replace(/\((.*?)\)/g, "") // Remove scientific names
            .replace(/[[\]"]/g, "")     // Remove extra JSON chars if any
            .trim(),
        }));
        setMeals(cleanedMeals);
      }

      setLoading(false);
    };

    fetchMeals();
  }, []);

  if (loading) return <p className="text-center text-lg font-semibold text-gray-600">Loading...</p>;
  if (error) return <p className="text-center text-lg text-red-500">Error: {error}</p>;

  const categorizedMeals = {
    breakfast: meals.filter((meal) => meal.meal_type.toLowerCase() === "breakfast"),
    lunch: meals.filter((meal) => meal.meal_type.toLowerCase() === "lunch"),
    dinner: meals.filter((meal) => meal.meal_type.toLowerCase() === "dinner"),
  };

  const scrollLeft = (type) => {
    if (mealRefs[type]?.current) {
      mealRefs[type].current.scrollBy({ left: -400, behavior: "smooth" });
    }
  };

  const scrollRight = (type) => {
    if (mealRefs[type]?.current) {
      mealRefs[type].current.scrollBy({ left: 400, behavior: "smooth" });
    }
  };

  return (
    <div className="min-h-screen bg-gradient-to-b from-white to-gray-100 p-6 flex flex-col items-center w-full">
      <h2 className="text-4xl font-bold text-gray-900 text-center mb-4">Your Personalized Diet Plan</h2>
      <p className="text-md text-gray-500 text-center mb-8 max-w-3xl font-light">
        A custom meal plan tailored to your nutrition needs. Keep track of your meals and maintain a balanced diet.
      </p>

      {["breakfast", "lunch", "dinner"].map((type) => (
        <div key={type} className="w-full max-w-7xl text-center mb-12 px-8">
          <h3 className="text-3xl font-bold text-blue-600 mb-4 capitalize">{type}</h3>

          <div className="relative">
            <button
              className="absolute left-0 top-1/2 transform -translate-y-1/2 p-3 bg-gray-200 rounded-full shadow-md hover:bg-gray-300 transition z-10"
              onClick={() => scrollLeft(type)}
            >
              <FaArrowLeft />
            </button>

            <div
              ref={mealRefs[type]}
              className="flex flex-nowrap gap-8 overflow-x-scroll snap-x snap-mandatory scroll-smooth px-6 hide-scrollbar justify-start"
            >
              {categorizedMeals[type].length > 0 ? (
                categorizedMeals[type].map((meal) => {
                  let nutritionInfo = {};
                  try {
                    nutritionInfo = JSON.parse(meal.nutrition_total);
                  } catch (error) {
                    console.error("Error parsing nutrition data:", error);
                  }

                  return (
                    <div
                      key={meal.id}
                      className="min-w-[350px] bg-white shadow-lg rounded-2xl p-6 snap-center border border-gray-200"
                    >
                      <h3 className="text-xl font-semibold text-gray-800">
                        {meal.meal_name}
                      </h3>
                      <p className="text-gray-600 mt-2">
                        <span className="font-bold">Ingredients:</span> {meal.meal_ingredients}
                      </p>
                      <p className="text-gray-600 mt-2 font-bold">Nutritional Info:</p>
                      <ul className="list-disc list-inside text-gray-600">
                        <li><strong>Carbs:</strong> {nutritionInfo.carbohydrate ? `${nutritionInfo.carbohydrate} g` : "N/A"}</li>
                        <li><strong>Protein:</strong> {nutritionInfo.protein ? `${nutritionInfo.protein} g` : "N/A"}</li>
                        <li><strong>Fat:</strong> {nutritionInfo.total_fat ? `${nutritionInfo.total_fat} g` : "N/A"}</li>
                        <li><strong>Iron:</strong> {nutritionInfo.iron_mg ? `${nutritionInfo.iron_mg} mg` : "N/A"}</li>
                      </ul>
                    </div>
                  );
                })
              ) : (
                <p className="text-gray-500">No meals found for {type}.</p>
              )}
            </div>

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

export default Personalized;
